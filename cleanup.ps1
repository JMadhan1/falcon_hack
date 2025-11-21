# Cleanup script for hackathon submission
Write-Host "🧹 Cleaning up repository for submission..." -ForegroundColor Cyan

# Remove duplicate test.py (we have predict.py)
if (Test-Path "test.py") {
    Remove-Item "test.py" -Force
    Write-Host "✓ Removed test.py (duplicate of predict.py)" -ForegroundColor Green
}

# Remove large model weights (they'll be in runs/detect/train/weights/)
if (Test-Path "yolov8n.pt") {
    Remove-Item "yolov8n.pt" -Force
    Write-Host "✓ Removed yolov8n.pt (pretrained weights)" -ForegroundColor Green
}

if (Test-Path "yolo11n.pt") {
    Remove-Item "yolo11n.pt" -Force
    Write-Host "✓ Removed yolo11n.pt (unused)" -ForegroundColor Green
}

# Remove dataset.zip (keep extracted dataset/)
if (Test-Path "dataset.zip") {
    Remove-Item "dataset.zip" -Force
    Write-Host "✓ Removed dataset.zip (4.5GB - keep extracted dataset/)" -ForegroundColor Green
}

# Remove temporary directories
if (Test-Path ".tmp.drivedownload") {
    Remove-Item ".tmp.drivedownload" -Recurse -Force
    Write-Host "✓ Removed .tmp.drivedownload/" -ForegroundColor Green
}

if (Test-Path ".tmp.driveupload") {
    Remove-Item ".tmp.driveupload" -Recurse -Force
    Write-Host "✓ Removed .tmp.driveupload/" -ForegroundColor Green
}

# Remove development/reference files
$devFiles = @(
    "COLAB_GUIDE.md",
    "QUICKSTART.md", 
    "REPORT_TEMPLATE.md",
    "HACKATHON_GUIDE.md",
    "setup.ps1",
    "setup_venv.ps1",
    "inspect_dataset.py",
    "Duality_AI_Hackathon_Training.ipynb"
)

foreach ($file in $devFiles) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "✓ Removed $file (development file)" -ForegroundColor Green
    }
}

Write-Host "`n✅ Cleanup complete!" -ForegroundColor Green
Write-Host "`n📦 Files to submit:" -ForegroundColor Cyan
Write-Host "  - train.py, validate.py, predict.py, failure_analysis.py"
Write-Host "  - data.yaml, requirements.txt"
Write-Host "  - README.md, HACKATHON_REPORT.md"
Write-Host "  - dataset/ (train, val, test)"
Write-Host "  - runs/ (training outputs)"
