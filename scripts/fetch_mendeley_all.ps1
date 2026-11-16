# Downloads the selected Mendeley datasets, extracts zips with tar.exe, deletes a zip only after a clean extract.
Set-Location (Split-Path $PSScriptRoot -Parent)
$sets = @(
    @{ name = 'dhan_shomadhan'; id = 'znsxdctwtt'; ver = 1 },
    @{ name = 'riceleafbd';     id = 'kx9rx8p2mz'; ver = 1 },
    @{ name = 'veg_leaf_bd';    id = 'n67gctmjyj'; ver = 3 },
    @{ name = 'wheat_bd';       id = '5gc7hwydwg'; ver = 1 }
)
foreach ($s in $sets) {
    $out = "data\raw\$($s.name)"
    Write-Output "=== $($s.name) ==="
    python scripts/download_mendeley.py $s.id $s.ver $out --workers 8
    Get-ChildItem $out -Recurse -Filter *.zip | ForEach-Object {
        $dest = Join-Path $_.DirectoryName ($_.BaseName)
        New-Item -ItemType Directory -Force $dest | Out-Null
        & tar.exe -xf $_.FullName -C $dest
        if ($LASTEXITCODE -eq 0) { Remove-Item $_.FullName -Confirm:$false; Write-Output "extracted + removed $($_.Name)" }
        else { Write-Output "EXTRACT FAILED $($_.Name), zip kept" }
    }
    Write-Output ("free GB: " + [math]::Round((Get-PSDrive C).Free / 1GB, 1))
}
Write-Output "ALL DONE"
