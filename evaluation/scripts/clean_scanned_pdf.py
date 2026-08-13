"""Cleans a badly-scanned PDF -- black scanner-bed background, a stray hand/
finger holding pages open, skew -- so pages read as plain white paper with
just the printed text, before OCR ever sees them.

Rasterizes each page, runs it through unpaper's auto page-content detection
(which finds the bright, roughly-rectangular page region and whites out
everything else -- no OCR bounding boxes needed, since the noise is a pixel-
level scanning artifact, not a text-layout one), reassembles the cleaned
pages into a PDF with img2pdf, then optionally re-runs ocrmypdf --force-ocr
on the result. Re-OCRing matters: OCR run against the noisy original
misreads the black/skin-tone background as spurious glyphs (confirmed on
page 1 of 9783406016127.pdf, which OCR'd as ISBN text interleaved with
"c3\"\" / ..." garbage from the hand), so a clean re-OCR is needed to get a
trustworthy text layer, not just a better-looking page image.

Also normalizes final page size: unpaper crops each page to its own detected
content, and since the source photographs weren't all taken at the exact
same zoom/distance, that leaves every page a slightly different pixel size
(115 distinct sizes across one 117-page book, observed on 9783406016127.pdf)
-- displayed inconsistently by any viewer that fits pages to a column width.
img2pdf's --pagesize centers each image on a shared physical page size
without resampling or cropping it (unlike unpaper's own --post-size, which
was tried first but empirically stretched narrow pages anisotropically to
fill the target box -- rejected for that reason). The shared size defaults
to the max width/height actually seen across the cleaned pages (--page-size
auto, guaranteeing no page ever needs shrinking) or can be pinned to a real
paper size like A5 (--page-size a5) to match the book's actual trim size --
in which case any page unpaper cropped wider than that target gets shrunk
to fit first (see shrink_to_fit), since img2pdf's centering alone doesn't
resample and would otherwise silently clip an oversized page.

Before any of that, each page also gets a horizontal-only ink trim (see
trim_horizontal) -- unpaper's mask-detection sometimes leaves a wide blank
margin on one side instead of cropping down to the true page edge, which
both mis-centers the page once normalized and wrongly shrinks it (the
erroneously wide crop looks bigger than it should to shrink_to_fit).

Wraps pdftoppm / unpaper / img2pdf / ImageMagick's `magick` / ocrmypdf as
external tools (developer-provided, not vendored) -- same convention as
pdfalto_runner.py's resolve_pdfalto_binary.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

STANDARD_PAGE_SIZES_MM: dict[str, tuple[float, float]] = {
    "a3": (297.0, 420.0),
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "letter": (215.9, 279.4),
    "legal": (215.9, 355.6),
}

_EXPLICIT_SIZE_RE = re.compile(r"^([\d.]+)(mm|cm|in)x([\d.]+)(mm|cm|in)$")
_MM_PER_UNIT = {"mm": 1.0, "cm": 10.0, "in": 25.4}

RASTER_MODE_PDFTOPPM_FLAGS: dict[str, list[str]] = {
    "color": [],
    "grayscale": ["-gray"],
}


def resolve_binary(cli_arg: str | None, env_var: str, default: str) -> str:
    """Resolves an external tool's binary path: explicit CLI flag, then the
    given environment variable, then the bare default name on PATH."""
    if cli_arg:
        return cli_arg
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    return default


def parse_page_size_mm(value: str) -> tuple[float, float] | None:
    """Parses --page-size into (width_mm, height_mm), or None for "auto"
    (derive the target from the max content size actually seen, so no page
    ever needs shrinking). Accepts a standard name (a5, a4, a3, letter,
    legal) or an explicit WIDTHxHEIGHT with mm/cm/in units, e.g.
    '148mmx210mm'."""
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized in STANDARD_PAGE_SIZES_MM:
        return STANDARD_PAGE_SIZES_MM[normalized]
    match = _EXPLICIT_SIZE_RE.match(normalized)
    if not match:
        raise ValueError(
            f"unrecognized --page-size value: {value!r} (expected 'auto', a "
            f"standard name like 'a5', or 'WIDTHxHEIGHT' with mm/cm/in units)"
        )
    width, width_unit, height, height_unit = match.groups()
    return float(width) * _MM_PER_UNIT[width_unit], float(height) * _MM_PER_UNIT[height_unit]


def rasterize_page(
    pdftoppm_bin: str, pdf_path: Path, page_num: int, dpi: int, raster_mode: str, out_dir: Path
) -> Path:
    """Rasterizes a single 1-based page to a .pgm/.ppm file in out_dir
    (format depends on raster_mode, one of RASTER_MODE_PDFTOPPM_FLAGS's
    keys) and returns its path. Uses a page-specific out_dir so pdftoppm's
    variable digit-padding in the output filename doesn't need to be
    predicted.

    Deliberately never rasterizes directly to 1-bit monochrome, even when
    the caller wants a monochrome *output*: unpaper's noise/blur/gray
    filters assume continuous grayscale and mistake a dithered monochrome
    page's halftone dots for scanning noise, erasing most of the real text
    (confirmed empirically on 9783406016127.pdf page 20 -- the noise-filter
    alone reported "deleted 74 clusters" on a page that came out nearly
    blank). Monochrome conversion has to happen after unpaper, on clean
    grayscale input -- see to_monochrome."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "raw"
    result = subprocess.run(
        [
            pdftoppm_bin,
            *RASTER_MODE_PDFTOPPM_FLAGS[raster_mode],
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(dpi),
            str(pdf_path),
            str(prefix),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed on page {page_num} of {pdf_path}: {result.stderr}")
    matches = sorted(out_dir.glob("raw*.p?m"))
    if not matches:
        raise RuntimeError(f"pdftoppm produced no output for page {page_num} of {pdf_path}")
    return matches[0]


def clean_page(unpaper_bin: str, raw_ppm: Path, out_dir: Path) -> Path:
    """Runs unpaper's default single-page auto-mask/deskew/border-detection
    on raw_ppm and returns the cleaned .ppm path. Default settings (no
    custom filter tuning) are enough: unpaper finds the bright page-content
    region around the sheet's center point and whites out the black
    background and any hand/finger intrusion around it."""
    cleaned_ppm = out_dir / "clean.ppm"
    result = subprocess.run(
        [unpaper_bin, "--overwrite", str(raw_ppm), str(cleaned_ppm)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not cleaned_ppm.exists():
        raise RuntimeError(f"unpaper failed on {raw_ppm}: {result.stderr}")
    return cleaned_ppm


def trim_horizontal(magick_bin: str, ppm_path: Path, out_dir: Path) -> Path:
    """Trims ppm_path to its ink content's left/right extent only, leaving
    the full vertical extent from unpaper's own crop untouched. Fixes a
    real defect found on 9783406016127.pdf: unpaper's mask-detection
    sometimes leaves a wide leftover blank margin on one side -- its
    blackfilter had correctly painted the scanning artifact (black
    background / hand) white in place, but the mask crop that determines
    the image's final width didn't shrink to match, leaving up to ~50% of
    the page as pure white padding INSIDE what unpaper called "content"
    (confirmed via ImageMagick's %@ trim geometry: e.g. a 2134px-wide crop
    whose actual ink only spanned pixels 1116-2051). Left uncorrected, this
    both mis-centers the page once normalized to a shared size (variable
    leftover margin -> variable final position) AND wrongly shrinks it
    (the erroneously wide crop makes shrink_to_fit downscale a page that
    didn't need it). The vertical extent is deliberately left alone here --
    unlike the horizontal one, it's consistently reliable (see
    shrink_to_fit's docstring and clean_scanned_pdf's module docstring), and
    2D-trimming it too would badly mis-center any lightly-filled page (e.g.
    a mostly-blank front-matter page would collapse to whatever tiny mark
    it has, centered in the middle of the sheet instead of near the top)."""
    width, height = read_pnm_size(ppm_path)
    result = subprocess.run(
        [magick_bin, str(ppm_path), "-fuzz", "5%", "-format", "%@", "info:"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"magick failed to measure trim box for {ppm_path}: {result.stderr}")
    match = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)", result.stdout.strip())
    if not match:
        raise RuntimeError(f"magick returned an unparseable trim box for {ppm_path}: {result.stdout!r}")
    trimmed_width, _, trim_x, _ = (int(g) for g in match.groups())
    trimmed_ppm = out_dir / "htrimmed.ppm"
    crop_result = subprocess.run(
        [magick_bin, str(ppm_path), "-crop", f"{trimmed_width}x{height}+{trim_x}+0", "+repage", str(trimmed_ppm)],
        capture_output=True,
        text=True,
    )
    if crop_result.returncode != 0 or not trimmed_ppm.exists():
        raise RuntimeError(f"magick failed to horizontally trim {ppm_path}: {crop_result.stderr}")
    return trimmed_ppm


def read_pnm_size(pnm_path: Path) -> tuple[int, int]:
    """Parses (width, height) from a PBM/PGM/PPM header, skipping comment
    lines, without needing Pillow."""
    with open(pnm_path, "rb") as f:
        magic = f.readline()
        if not magic.startswith(b"P"):
            raise ValueError(f"{pnm_path} is not a valid PNM file")
        tokens: list[bytes] = []
        while len(tokens) < 2:
            line = f.readline()
            if not line:
                raise ValueError(f"{pnm_path}: truncated PNM header")
            tokens.extend(line.split(b"#", 1)[0].split())
    return int(tokens[0]), int(tokens[1])


def shrink_to_fit(magick_bin: str, ppm_path: Path, target_size_px: tuple[int, int], out_dir: Path) -> Path:
    """Shrinks ppm_path to fit within target_size_px (preserving aspect
    ratio, via ImageMagick's '>' resize flag, a no-op if it already fits)
    and returns the resulting path. Needed because img2pdf's --pagesize
    only centers an image on a differently-sized page -- it does not
    resample -- so a page bigger than a *fixed* target (e.g. a real A5
    target smaller than some unpaper-cropped page) would otherwise overflow
    the page and get silently clipped by anything that rasterizes the PDF
    (confirmed empirically: img2pdf keeps the source image at its native
    pixel size regardless of --imgsize/--pagesize, so an oversized page fed
    to it directly loses its far edge -- e.g. a page's rotated spine text --
    once rendered, rather than being scaled down)."""
    width, height = read_pnm_size(ppm_path)
    if width <= target_size_px[0] and height <= target_size_px[1]:
        return ppm_path
    shrunk_ppm = out_dir / "shrunk.ppm"
    result = subprocess.run(
        [magick_bin, str(ppm_path), "-resize", f"{target_size_px[0]}x{target_size_px[1]}>", str(shrunk_ppm)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not shrunk_ppm.exists():
        raise RuntimeError(f"magick failed to shrink {ppm_path}: {result.stderr}")
    return shrunk_ppm


def to_monochrome(magick_bin: str, ppm_path: Path, out_dir: Path) -> Path:
    """Converts a clean grayscale ppm_path to 1-bit black/white via an
    Otsu-optimal global threshold (no error-diffusion dithering, which
    would reintroduce the halftone-vs-noise-filter conflict rasterize_page
    avoids -- this must only ever run on already-unpaper-cleaned, already-
    shrunk grayscale input, never before it)."""
    mono_pbm = out_dir / "mono.pbm"
    result = subprocess.run(
        [magick_bin, str(ppm_path), "-auto-threshold", "otsu", str(mono_pbm)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not mono_pbm.exists():
        raise RuntimeError(f"magick failed to threshold {ppm_path} to monochrome: {result.stderr}")
    return mono_pbm


def assemble_pdf(
    img2pdf_bin: str,
    page_images: list[Path],
    dpi: int,
    target_size_px: tuple[int, int] | None,
    out_pdf: Path,
) -> None:
    """Assembles page_images into out_pdf. Always passes --imgsize so pages
    get the correct physical size at the given dpi (img2pdf otherwise
    silently assumes 96dpi for dpi-less PNM input, inflating every page's
    declared point size by a constant ~3x factor). When target_size_px is
    given, also passes --pagesize so every page shares that physical size,
    with each image centered on it unscaled -- img2pdf only centers in this
    mode, it doesn't resample, so smaller pages just get white padding.
    Callers needing a *fixed* target smaller than some page's native size
    must pre-shrink that page with shrink_to_fit first, since img2pdf alone
    won't downscale it (see shrink_to_fit's docstring)."""
    args = [img2pdf_bin, "--imgsize", f"{dpi}dpi"]
    if target_size_px is not None:
        target_w_in = target_size_px[0] / dpi
        target_h_in = target_size_px[1] / dpi
        args += ["--pagesize", f"{target_w_in:.4f}inx{target_h_in:.4f}in"]
    args += [str(p) for p in page_images]
    args += ["-o", str(out_pdf)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 or not out_pdf.exists():
        raise RuntimeError(f"img2pdf failed to assemble {len(page_images)} pages: {result.stderr}")


def reocr_pdf(
    ocrmypdf_bin: str, lang: str, optimize: int, jbig2_lossy: bool, in_pdf: Path, out_pdf: Path
) -> None:
    """Re-OCRs in_pdf and, via ocrmypdf's own post-OCR optimizer, recompresses
    its images: -O 1 (the default, and ocrmypdf's own default too) applies
    only safe lossless recompression -- already using JBIG2 for our
    monochrome pages (confirmed via pdfimages -list: 'enc jbig2', jbig2enc is
    installed), which is why monochrome+A5 output is already far smaller
    than the unoptimized img2pdf assembly. -O 2/3 additionally allow lossy
    JPEG recompression of grayscale/color images; jbig2_lossy separately
    enables JBIG2's lossy symbol-substitution mode (smaller, but can
    occasionally confuse similar-looking glyphs) for monochrome images --
    orthogonal to -O, so it's passed regardless of the chosen level."""
    args = [ocrmypdf_bin, "--force-ocr", "-l", lang, "-O", str(optimize)]
    if jbig2_lossy:
        args.append("--jbig2-lossy")
    args += [str(in_pdf), str(out_pdf)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 or not out_pdf.exists():
        raise RuntimeError(f"ocrmypdf failed on {in_pdf}: {result.stderr}")


def clean_scanned_pdf(
    input_pdf: Path,
    output_pdf: Path,
    dpi: int,
    start_page: int,
    end_page: int | None,
    ocr_lang: str | None,
    normalize_page_size: bool,
    page_size_mm: tuple[float, float] | None,
    color_mode: str,
    optimize: int,
    jbig2_lossy: bool,
    pdftoppm_bin: str,
    unpaper_bin: str,
    img2pdf_bin: str,
    magick_bin: str,
    ocrmypdf_bin: str,
    work_dir: Path,
) -> None:
    total_pages = len(PdfReader(str(input_pdf)).pages)
    last_page = end_page if end_page is not None else total_pages
    if last_page > total_pages:
        raise ValueError(f"end_page {last_page} exceeds {input_pdf}'s {total_pages} pages")

    raster_mode = "color" if color_mode == "color" else "grayscale"
    cleaned_pages: list[Path] = []
    page_sizes: list[tuple[int, int]] = []
    for page_num in range(start_page, last_page + 1):
        page_dir = work_dir / f"page_{page_num:04d}"
        raw_ppm = rasterize_page(pdftoppm_bin, input_pdf, page_num, dpi, raster_mode, page_dir)
        cleaned_ppm = clean_page(unpaper_bin, raw_ppm, page_dir)
        cleaned_ppm = trim_horizontal(magick_bin, cleaned_ppm, page_dir)
        cleaned_pages.append(cleaned_ppm)
        page_sizes.append(read_pnm_size(cleaned_ppm))
        print(f"cleaned page {page_num}/{last_page}", file=sys.stderr)

    target_size_px = None
    if normalize_page_size:
        if page_size_mm is not None:
            target_size_px = (round(page_size_mm[0] / 25.4 * dpi), round(page_size_mm[1] / 25.4 * dpi))
        else:
            target_size_px = (max(w for w, _ in page_sizes), max(h for _, h in page_sizes))
        print(f"normalizing to shared page size {target_size_px[0]}x{target_size_px[1]}px", file=sys.stderr)
        cleaned_pages = [shrink_to_fit(magick_bin, p, target_size_px, p.parent) for p in cleaned_pages]

    if color_mode == "monochrome":
        cleaned_pages = [to_monochrome(magick_bin, p, p.parent) for p in cleaned_pages]

    if ocr_lang:
        raster_pdf = work_dir / "raster.pdf"
        assemble_pdf(img2pdf_bin, cleaned_pages, dpi, target_size_px, raster_pdf)
        print("re-running OCR on cleaned pages...", file=sys.stderr)
        reocr_pdf(ocrmypdf_bin, ocr_lang, optimize, jbig2_lossy, raster_pdf, output_pdf)
    else:
        assemble_pdf(img2pdf_bin, cleaned_pages, dpi, target_size_px, output_pdf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a badly-scanned PDF (black scanner background, "
        "stray hand/fingers, skew) by re-rasterizing and running unpaper's "
        "auto page-content detection, then optionally re-OCR the result."
    )
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization resolution (default: 300, good for re-OCR).")
    parser.add_argument("--start-page", type=int, default=1, help="1-based first page to process (default: 1).")
    parser.add_argument("--end-page", type=int, default=None, help="1-based last page to process (default: last page).")
    parser.add_argument(
        "--ocr-lang",
        default=None,
        help="Re-OCR the cleaned pages with ocrmypdf --force-ocr using this "
        "tesseract language code (e.g. 'deu'). Omit to skip re-OCR and "
        "produce an image-only PDF (no text layer).",
    )
    parser.add_argument(
        "--no-normalize-page-size",
        dest="normalize_page_size",
        action="store_false",
        help="Skip normalizing to a shared page size -- leave every page's "
        "PDF page size matching its own unpaper-detected content size.",
    )
    parser.add_argument(
        "--page-size",
        default="auto",
        help="Target page size to normalize to (ignored if "
        "--no-normalize-page-size is given): 'auto' (default) uses the max "
        "width/height actually seen across the processed pages, so no page "
        "ever needs shrinking; a standard name ('a5', 'a4', 'a3', 'letter', "
        "'legal'); or an explicit WIDTHxHEIGHT with mm/cm/in units (e.g. "
        "'148mmx210mm'). Pages larger than the target in either dimension "
        "are shrunk to fit it first (preserving aspect ratio, via "
        "ImageMagick); every page is then centered on the shared target "
        "size unscaled.",
    )
    parser.add_argument(
        "--color-mode",
        choices=["grayscale", "monochrome", "color"],
        default="grayscale",
        help="Rasterization color mode (default: grayscale -- a "
        "black-and-white book scan doesn't need full color's 3x file "
        "size). 'monochrome' renders 1-bit black/white (smaller still, but "
        "can lose antialiasing detail that helps OCR). 'color' keeps the "
        "original RGB.",
    )
    parser.add_argument(
        "--optimize",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Post-OCR optimization level, passed to ocrmypdf's -O (only "
        "takes effect when --ocr-lang is given, since that's what runs "
        "ocrmypdf): 1 (default) applies only safe lossless recompression "
        "-- already using JBIG2 for monochrome pages if jbig2enc is "
        "installed. 2 and 3 additionally allow lossy JPEG recompression of "
        "grayscale/color images (smaller, minor quality loss); irrelevant "
        "for monochrome pages on their own -- pair with --jbig2-lossy for "
        "smaller monochrome output too.",
    )
    parser.add_argument(
        "--jbig2-lossy",
        action="store_true",
        help="Allow ocrmypdf's JBIG2 encoder to use lossy symbol "
        "substitution for monochrome images -- meaningfully smaller than "
        "the lossless JBIG2 --optimize already uses, but can occasionally "
        "substitute a similar-looking glyph for another. Only takes effect "
        "when --ocr-lang is given.",
    )
    parser.add_argument("--pdftoppm-bin", default=None)
    parser.add_argument("--unpaper-bin", default=None)
    parser.add_argument("--img2pdf-bin", default=None)
    parser.add_argument("--magick-bin", default=None)
    parser.add_argument("--ocrmypdf-bin", default=None)
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Don't delete the temporary per-page working directory; print its path for inspection.",
    )
    args = parser.parse_args()
    page_size_mm = parse_page_size_mm(args.page_size)

    pdftoppm_bin = resolve_binary(args.pdftoppm_bin, "PDFTOPPM_BIN", "pdftoppm")
    unpaper_bin = resolve_binary(args.unpaper_bin, "UNPAPER_BIN", "unpaper")
    img2pdf_bin = resolve_binary(args.img2pdf_bin, "IMG2PDF_BIN", "img2pdf")
    magick_bin = resolve_binary(args.magick_bin, "MAGICK_BIN", "magick")
    ocrmypdf_bin = resolve_binary(args.ocrmypdf_bin, "OCRMYPDF_BIN", "ocrmypdf")

    work_dir = Path(tempfile.mkdtemp(prefix="clean_scanned_pdf_"))
    try:
        clean_scanned_pdf(
            args.input_pdf,
            args.output_pdf,
            args.dpi,
            args.start_page,
            args.end_page,
            args.ocr_lang,
            args.normalize_page_size,
            page_size_mm,
            args.color_mode,
            args.optimize,
            args.jbig2_lossy,
            pdftoppm_bin,
            unpaper_bin,
            img2pdf_bin,
            magick_bin,
            ocrmypdf_bin,
            work_dir,
        )
    finally:
        if args.keep_workdir:
            print(f"working directory kept at: {work_dir}", file=sys.stderr)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
