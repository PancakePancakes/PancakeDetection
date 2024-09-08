# 获取当前工作目录
$currentPath = Get-Location

# 获取当前的 PYTHONPATH，如果没有则为空字符串
$currentPythonPath = [System.Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')

# 添加当前工作目录到 PYTHONPATH
if ($currentPythonPath) {
    $newPythonPath = "$currentPath;$currentPythonPath"
} else {
    $newPythonPath = "$currentPath"
}

# 设置新的 PYTHONPATH
[System.Environment]::SetEnvironmentVariable('PYTHONPATH', $newPythonPath, 'Process')

# 输出新的 PYTHONPATH
Write-Host "Updated PYTHONPATH: $newPythonPath"
