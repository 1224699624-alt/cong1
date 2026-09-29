$ErrorActionPreference = 'Stop'
$miktex = Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64'
if (Test-Path -LiteralPath $miktex) {
    $env:Path = "$miktex;$env:Path"
}
Push-Location $PSScriptRoot
try {
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
} finally {
    Pop-Location
}
