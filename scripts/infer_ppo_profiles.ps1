$python = 'E:\Programs\anaconda3\envs\zmd_pyqt\python.exe'
$projectRoot = 'E:\sjtu\医疗机器人\血管机器人\ZMD_PY_RL'
$batchRoot = Join-Path $projectRoot 'projects\cache\rl_batch\20260720_110726'
$inferScript = Join-Path $projectRoot 'scripts\infer_rl_cli.py'

# Follow-up experiment only; keep the existing profile CSV files untouched.
$profiles = @('P90', 'P95')

$routes = @(
    @{ Name = 'entry1_exit1'; Entry = 0; Exit = 0 }
    @{ Name = 'entry1_exit2'; Entry = 0; Exit = 1 }
    @{ Name = 'entry1_exit3'; Entry = 0; Exit = 2 }
    @{ Name = 'entry2_exit1'; Entry = 1; Exit = 0 }
    @{ Name = 'entry2_exit2'; Entry = 1; Exit = 1 }
    @{ Name = 'entry2_exit3'; Entry = 1; Exit = 2 }
)

foreach ($route in $routes) {
    $routeName = $route.Name
    $entryIdx = $route.Entry
    $exitIdx = $route.Exit
    $routeRoot = Join-Path $batchRoot "ppo\$routeName"

    $checkpoints = @(
        Get-ChildItem `
            -Path "$routeRoot\repeat_*\ppo_*\best_model.zip" `
            -File
    )

    Write-Host ''
    Write-Host '=================================================='
    Write-Host "路线: $routeName；模型数量: $($checkpoints.Count)"
    Write-Host '=================================================='

    foreach ($checkpoint in $checkpoints) {
        $ckpt = $checkpoint.FullName
        $repeatName = $checkpoint.Directory.Parent.Name

        # CSV 直接放入 repeat_XX
        $outputDir = Join-Path $routeRoot $repeatName

        if (-not (Test-Path -LiteralPath $outputDir)) {
            throw "输出目录不存在: $outputDir"
        }

        foreach ($profile in $profiles) {
            $output = Join-Path `
                $outputDir `
                "${routeName}_${repeatName}_${profile}.csv"

            Write-Host "运行: $routeName / $repeatName / $profile"
            Write-Host "输出: $output"

            & $python $inferScript $ckpt `
                --algo PPO `
                --entry $entryIdx `
                --exit $exitIdx `
                --success-progress 0.99 `
                --max-steps 1200 `
                --speed-profile $profile `
                --tick-ms 90 `
                --output $output

            if ($LASTEXITCODE -ne 0) {
                Write-Warning "推理失败: $routeName / $repeatName / $profile"
            }
        }
    }
}
