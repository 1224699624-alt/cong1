param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [string]$WebhookUrl = "",
    [string]$Title = "Workflow 2"
)

$ErrorActionPreference = "Stop"

if (-not $WebhookUrl) {
    $WebhookUrl = $env:WORKFLOW2_WEBHOOK_URL
}

if (-not $WebhookUrl) {
    Write-Host "[INFO] No webhook configured. Skipping notification."
    exit 0
}

$Body = @{
    msg_type = "text"
    content = @{
        text = "$Title`n$Message"
    }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
    -Uri $WebhookUrl `
    -Method Post `
    -ContentType "application/json" `
    -Body $Body | Out-Null

Write-Host "[OK] Notification sent."
