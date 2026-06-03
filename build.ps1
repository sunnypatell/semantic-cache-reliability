# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
#
# Compile paper/main.tex quickly. The project lives in a OneDrive-synced folder where
# every font/aux file operation is scanned and ~75x slower (116s vs 1.5s observed), so
# we copy the sources to a local scratch directory, build there with latexmk (using the
# Perl that ships with Git for Windows), and copy the finished PDF back to paper/.
#
# Usage:  powershell -File build.ps1 [-Clean]
param([switch]$Clean)
# latexmk writes progress notes to stderr; with Stop + redirection PowerShell would treat
# the first such line as fatal and abort before biber runs. Keep going on non-fatal output.
$ErrorActionPreference = 'Continue'

$git = 'C:\Program Files\Git\usr\bin'
if (Test-Path (Join-Path $git 'perl.exe')) { $env:PATH = "$git;$env:PATH" }

$src = Join-Path $PSScriptRoot 'paper'
$build = Join-Path $env:LOCALAPPDATA 'cacherel-build'
New-Item -ItemType Directory -Force $build | Out-Null

Copy-Item (Join-Path $src 'main.tex') $build -Force
Copy-Item (Join-Path $src 'references.bib') $build -Force
$figsrc = Join-Path $src 'figures'
if (Test-Path $figsrc) {
  New-Item -ItemType Directory -Force (Join-Path $build 'figures') | Out-Null
  Copy-Item (Join-Path $figsrc '*') (Join-Path $build 'figures') -Recurse -Force -ErrorAction SilentlyContinue
}

Push-Location $build
if ($Clean) { cmd /c "latexmk -C main.tex > nul 2>&1" }
$sw = [System.Diagnostics.Stopwatch]::StartNew()
cmd /c "latexmk -pdf -interaction=nonstopmode -file-line-error main.tex > latexmk.run.txt 2>&1"
$code = $LASTEXITCODE
$sw.Stop()
Pop-Location

if (Test-Path (Join-Path $build 'main.pdf')) {
  Copy-Item (Join-Path $build 'main.pdf') (Join-Path $src 'main.pdf') -Force
  $pages = (Select-String -Path (Join-Path $build 'main.log') -Pattern 'Output written on main.pdf \((\d+) page').Matches.Groups[1].Value
  Write-Output ("build OK (exit=$code) in {0:N1}s, $pages pages -> paper/main.pdf" -f $sw.Elapsed.TotalSeconds)
  $warn = Select-String -Path (Join-Path $build 'main.log') -Pattern 'Overfull|Underfull|undefined reference|LaTeX Warning' | Select-Object -Last 8
  if ($warn) { Write-Output "warnings:"; $warn | ForEach-Object { $_.Line } }
} else {
  Write-Output "build FAILED (exit=$code); tail:"
  Get-Content (Join-Path $build 'latexmk.run.txt') -Tail 20
}