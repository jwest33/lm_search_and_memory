# SearXNG + Valkey Setup Script for Windows
# Run this in PowerShell as Administrator

param(
    [switch]$GenerateSecretKey,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Logs,
    [switch]$Test,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Generate-SecretKey {
    Write-ColorOutput Yellow "Generating secure secret key..."
    
    $randomBytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::new()
    $rng.GetBytes($randomBytes)
    $secretKey = -join ($randomBytes | ForEach-Object { "{0:x2}" -f $_ })
    $rng.Dispose()
    
    $settingsPath = Join-Path $ScriptDir "searxng\settings.yml"
    
    if (Test-Path $settingsPath) {
        $content = Get-Content $settingsPath -Raw
        $content = $content -replace 'secret_key: "REPLACE_WITH_SECURE_KEY_openssl_rand_hex_32"', "secret_key: `"$secretKey`""
        $content = $content -replace 'secret_key: "[a-f0-9]{64}"', "secret_key: `"$secretKey`""
        Set-Content $settingsPath $content -NoNewline
        Write-ColorOutput Green "Secret key generated and saved to settings.yml"
    } else {
        Write-ColorOutput Red "settings.yml not found at: $settingsPath"
        Write-Output "Secret key for manual use: $secretKey"
    }
}

function Start-SearXNG {
    Write-ColorOutput Yellow "Starting SearXNG and Valkey containers..."
    Push-Location $ScriptDir
    try {
        docker compose up -d
        Write-ColorOutput Green "Containers started successfully!"
        Write-Output ""
        Write-Output "SearXNG is now available at: http://localhost:8080"
        Write-Output "JSON API endpoint: http://localhost:8080/search?q=test&format=json"
    } finally {
        Pop-Location
    }
}

function Stop-SearXNG {
    Write-ColorOutput Yellow "Stopping SearXNG and Valkey containers..."
    Push-Location $ScriptDir
    try {
        docker compose down
        Write-ColorOutput Green "Containers stopped successfully!"
    } finally {
        Pop-Location
    }
}

function Restart-SearXNG {
    Write-ColorOutput Yellow "Restarting SearXNG and Valkey containers..."
    Push-Location $ScriptDir
    try {
        docker compose restart
        Write-ColorOutput Green "Containers restarted successfully!"
    } finally {
        Pop-Location
    }
}

function Show-Logs {
    Push-Location $ScriptDir
    try {
        docker compose logs -f
    } finally {
        Pop-Location
    }
}

function Test-SearXNG {
    Write-ColorOutput Yellow "Testing SearXNG JSON API..."
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/search?q=test&format=json" -UseBasicParsing
        $json = $response.Content | ConvertFrom-Json
        
        Write-ColorOutput Green "SUCCESS: SearXNG is responding!"
        Write-Output ""
        Write-Output "Response status: $($response.StatusCode)"
        Write-Output "Number of results: $($json.results.Count)"
        
        if ($json.results.Count -gt 0) {
            Write-Output ""
            Write-Output "First result:"
            Write-Output "  Title: $($json.results[0].title)"
            Write-Output "  URL: $($json.results[0].url)"
        }
    } catch {
        Write-ColorOutput Red "FAILED: Could not connect to SearXNG"
        Write-Output "Error: $_"
        Write-Output ""
        Write-Output "Make sure:"
        Write-Output "1. Docker Desktop is running"
        Write-Output "2. Containers are started (run with -Start)"
        Write-Output "3. Port 8080 is not in use by another application"
    }
}

function Show-Status {
    Write-ColorOutput Yellow "Container Status:"
    Push-Location $ScriptDir
    try {
        docker compose ps
    } finally {
        Pop-Location
    }
}

# Main execution
if ($GenerateSecretKey) { Generate-SecretKey }
elseif ($Start) { Start-SearXNG }
elseif ($Stop) { Stop-SearXNG }
elseif ($Restart) { Restart-SearXNG }
elseif ($Logs) { Show-Logs }
elseif ($Test) { Test-SearXNG }
elseif ($Status) { Show-Status }
else {
    Write-Output @"
SearXNG + Valkey Management Script

Usage: .\setup.ps1 [option]

Options:
  -GenerateSecretKey    Generate and save a secure secret key
  -Start                Start the containers
  -Stop                 Stop the containers
  -Restart              Restart the containers
  -Logs                 Follow container logs
  -Test                 Test the JSON API endpoint
  -Status               Show container status

First-time setup:
  1. .\setup.ps1 -GenerateSecretKey
  2. .\setup.ps1 -Start
  3. .\setup.ps1 -Test

"@
}
