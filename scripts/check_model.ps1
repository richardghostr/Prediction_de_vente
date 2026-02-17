# Run from project root: .\scripts\check_model.ps1
# Confirms the trained model exists so Docker can use it (Option A).
$artifacts = "models\artifacts"
$pkls = Get-ChildItem -Path $artifacts -Filter "model_*_baseline.pkl" -ErrorAction SilentlyContinue
if ($pkls) {
    Write-Host "OK: Found $($pkls.Count) model(s) in $artifacts"
    $pkls | ForEach-Object { Write-Host "  - $($_.Name)" }
    Write-Host "You can start Docker: docker-compose up --build"
} else {
    Write-Host "NO MODEL in $artifacts - predictions will be flat (naive)."
    Write-Host "Run first: python -m src.models.train"
    exit 1
}
