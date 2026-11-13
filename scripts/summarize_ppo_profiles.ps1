$batchRoot = 'E:\sjtu\医疗机器人\血管机器人\ZMD_PY_RL\projects\cache\rl_batch\20260720_110726'

$summaryFiles = @(
    Get-ChildItem `
        -Path "$batchRoot\ppo\entry*_exit*\repeat_*\*_summary.csv" `
        -File
)

Write-Host "找到 summary 文件数量: $($summaryFiles.Count)"

if ($summaryFiles.Count -eq 0) {
    throw "未找到 summary CSV，请检查推理输出目录。"
}

$results = $summaryFiles | ForEach-Object {
    Import-Csv -LiteralPath $_.FullName
}

$results |
    Group-Object route, speed_profile |
    ForEach-Object {
        $group = $_.Group
    [PSCustomObject]@{
        Route            = $group[0].route
        Profile          = $group[0].speed_profile
        Runs             = $group.Count
        SuccessCount     = (
            $group |
            Where-Object { [int]$_.success -eq 1 }
        ).Count
        SuccessRate      = [math]::Round(
            ($group |
                Measure-Object success -Average
            ).Average,
            4
        )
        MeanSteps        = [math]::Round(
            ($group |
                Measure-Object steps -Average
            ).Average,
            2
        )
        MeanTimeS        = [math]::Round(
            ($group |
                Measure-Object estimated_execution_time_s -Average
            ).Average,
            2
        )
        MaxTimeS         = [math]::Round(
            ($group |
                Measure-Object estimated_execution_time_s -Maximum
            ).Maximum,
            2
        )
        MeanProgress     = [math]::Round(
            ($group |
                Measure-Object final_progress -Average
            ).Average,
            4
        )
        TotalRotations   = (
            $group |
            Measure-Object rotation_count -Sum
        ).Sum
        MeanBlockedRate  = [math]::Round(
            ($group |
                Measure-Object blocked_advance_rate -Average
            ).Average,
            4
        )
    }
} |
Sort-Object Route, Profile |
Format-Table -AutoSize