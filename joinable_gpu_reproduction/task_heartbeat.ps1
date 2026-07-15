param(
    [Parameter(Mandatory = $true)]
    [datetime]$TaskStart,

    [string]$MarkerPath = "C:\Users\11049\Desktop\Codex.txt",

    [int]$IntervalSeconds = 300
)

$ErrorActionPreference = "Stop"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function ConvertFrom-CodePoints {
    param([int[]]$CodePoints)

    return -join ($CodePoints | ForEach-Object { [char]$_ })
}

function Format-Elapsed {
    param([timespan]$Elapsed)

    $day = ConvertFrom-CodePoints @(0x5929)
    $hour = ConvertFrom-CodePoints @(0x5C0F, 0x65F6)
    $minute = ConvertFrom-CodePoints @(0x5206, 0x949F)
    $second = ConvertFrom-CodePoints @(0x79D2)

    if ($Elapsed.Days -gt 0) {
        return "{0}{1} {2:00}{3} {4:00}{5} {6:00}{7}" -f `
            $Elapsed.Days, $day, $Elapsed.Hours, $hour, `
            $Elapsed.Minutes, $minute, $Elapsed.Seconds, $second
    }

    return "{0:00}{1} {2:00}{3} {4:00}{5}" -f `
        [math]::Floor($Elapsed.TotalHours), $hour, `
        $Elapsed.Minutes, $minute, $Elapsed.Seconds, $second
}

function Write-MarkerAtomically {
    param(
        [string]$Path,
        [string]$Text
    )

    $directory = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $directory)) {
        [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    }

    $temporaryPath = "$Path.tmp.$PID"
    [System.IO.File]::WriteAllText($temporaryPath, $Text, $utf8NoBom)
    Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
}

while ($true) {
    $now = Get-Date
    $elapsed = $now - $TaskStart
    $status = ConvertFrom-CodePoints @(0x6B63, 0x5728, 0x8FDB, 0x884C, 0x4E2D)
    $updatedAt = ConvertFrom-CodePoints @(0x66F4, 0x65B0, 0x65F6, 0x95F4)
    $taskElapsed = ConvertFrom-CodePoints @(
        0x672C, 0x6B21, 0x957F, 0x4EFB, 0x52A1, 0x5DF2, 0x8FD0, 0x884C
    )
    $heartbeatProcess = ConvertFrom-CodePoints @(0x5FC3, 0x8DF3, 0x8FDB, 0x7A0B)
    $colon = ConvertFrom-CodePoints @(0xFF1A)
    $leftParenthesis = ConvertFrom-CodePoints @(0xFF08)
    $rightParenthesis = ConvertFrom-CodePoints @(0xFF09)
    $every = ConvertFrom-CodePoints @(0x6BCF)
    $second = ConvertFrom-CodePoints @(0x79D2)
    $update = ConvertFrom-CodePoints @(0x66F4, 0x65B0)

    $text = @(
        $status
        "${updatedAt}${colon}$($now.ToString('yyyy-MM-dd HH:mm:ss'))"
        "${taskElapsed}${colon}$(Format-Elapsed -Elapsed $elapsed)"
        "${heartbeatProcess}${colon}${PID}${leftParenthesis}${every} $IntervalSeconds ${second}${update}${rightParenthesis}"
    ) -join [Environment]::NewLine

    Write-MarkerAtomically -Path $MarkerPath -Text $text
    Start-Sleep -Seconds $IntervalSeconds
}
