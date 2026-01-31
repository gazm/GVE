# backend/architect/test_generate.ps1
# PowerShell script to test the AI generation endpoint

$body = @{
    prompt = "A weathered wooden barrel with iron bands"
} | ConvertTo-Json

$headers = @{
    "Content-Type" = "application/json"
}

try {
    Write-Host "[*] Sending generation request..." -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8888/api/generate" `
        -Method Post `
        -Headers $headers `
        -Body $body
    
    Write-Host "[+] Request accepted!" -ForegroundColor Green
    Write-Host "Job ID: $($response.job_id)" -ForegroundColor Yellow
    Write-Host "Status: $($response.status)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[*] Polling for status..." -ForegroundColor Cyan
    
    $jobId = $response.job_id
    $maxAttempts = 300  # 10 minutes (300 * 2 seconds)
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        Start-Sleep -Seconds 2
        try {
            $status = Invoke-RestMethod -Uri "http://localhost:8888/api/generate/status/$jobId" -Method Get
            
            Write-Host "  Status: $($status.status)" -ForegroundColor Gray
            
            if ($status.status -eq "completed") {
                Write-Host ""
                Write-Host "[+] Generation complete!" -ForegroundColor Green
                Write-Host "Asset ID: $($status.asset_id)" -ForegroundColor Yellow
                if ($status.result) {
                    Write-Host "Time: $($status.result.generation_time_sec) seconds" -ForegroundColor Yellow
                    Write-Host "Confidence: $($status.result.confidence)" -ForegroundColor Yellow
                }
                break
            } elseif ($status.status -eq "failed") {
                Write-Host ""
                Write-Host "[-] Generation failed!" -ForegroundColor Red
                Write-Host "Error: $($status.error)" -ForegroundColor Red
                break
            }
        } catch {
            Write-Host "  [!] Error checking status: $_" -ForegroundColor Yellow
        }
        
        $attempt++
        
        # Show progress every 30 attempts (1 minute)
        if ($attempt % 30 -eq 0) {
            Write-Host "  [*] Still running... ($attempt attempts, ~$([math]::Round($attempt * 2 / 60)) minutes)" -ForegroundColor Cyan
        }
    }
    
    if ($attempt -ge $maxAttempts) {
        Write-Host ""
        Write-Host "[!] Timeout waiting for completion after $maxAttempts attempts (~$([math]::Round($maxAttempts * 2 / 60)) minutes)" -ForegroundColor Yellow
        Write-Host "[*] Check server console for errors or use: Invoke-RestMethod -Uri 'http://localhost:8888/api/generate/debug/$jobId' -Method Get" -ForegroundColor Cyan
    }
    
} catch {
    Write-Host "[-] Error: $_" -ForegroundColor Red
    Write-Host 'Make sure the server is running: uvicorn src.api:app --reload' -ForegroundColor Yellow
}
