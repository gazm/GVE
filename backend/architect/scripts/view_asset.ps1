# backend/architect/view_asset.ps1
# Helper script to check asset status and provide load instructions

param(
    [string]$AssetId = "6978364185beef42fa4f3ae3"
)

$baseUrl = "http://localhost:8888"

Write-Host "[*] Checking asset: $AssetId" -ForegroundColor Cyan

# Check if asset exists
try {
    $asset = Invoke-RestMethod -Uri "$baseUrl/api/assets/$AssetId" -Method Get
    Write-Host "[+] Asset found: $($asset.name)" -ForegroundColor Green
} catch {
    Write-Host "[-] Asset not found: $_" -ForegroundColor Red
    exit 1
}

# Check if binary exists (compiled)
try {
    $binary = Invoke-RestMethod -Uri "$baseUrl/api/assets/$AssetId/binary" -Method Get -ErrorAction Stop
    Write-Host "[+] Binary file exists - Asset is compiled!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To view in Forge UI:" -ForegroundColor Yellow
    Write-Host "1. Open Forge UI (http://localhost:8080)" -ForegroundColor White
    Write-Host "2. Open browser console (F12)" -ForegroundColor White
    Write-Host "3. Run: window.load_asset('/api/assets/$AssetId/binary', 1)" -ForegroundColor Cyan
} catch {
    Write-Host "[-] Binary not found - Asset needs compilation" -ForegroundColor Yellow
    Write-Host "[*] Triggering compilation..." -ForegroundColor Cyan
    
    try {
        $compileJob = Invoke-RestMethod -Uri "$baseUrl/api/compile/$AssetId" -Method Post -ContentType "application/json" -Body '{}'
        Write-Host "[+] Compilation queued: $($compileJob.job_id)" -ForegroundColor Green
        Write-Host "[*] Wait a few seconds, then check again with: .\view_asset.ps1 -AssetId $AssetId" -ForegroundColor Yellow
    } catch {
        Write-Host "[-] Failed to trigger compilation: $_" -ForegroundColor Red
    }
}
