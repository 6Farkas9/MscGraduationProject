# 激活Anaconda环境并依次运行Python脚本的PowerShell脚本
# 文件名: run_scripts.ps1

# 1. 设置变量（修改这些值为你实际需要的）
$condaEnvName = "pyt240cu124"  # 替换为你的Anaconda环境名
$pythonScript1 = "D:\Desktop\GraduationDesign\GraduationDesign\DeepLearning\KT\train.py"# 第一个要运行的Python脚本
$pythonScript2 = "D:\Desktop\GraduationDesign\GraduationDesign\DeepLearning\CD\train.py"# 第二个要运行的Python脚本
$pythonScript3 = "D:\Desktop\GraduationDesign\GraduationDesign\DeepLearning\RR\train.py"# 第三个要运行的Python脚本

# 2. 激活Anaconda环境
Write-Host "正在激活Anaconda环境: $condaEnvName"
conda activate $condaEnvName
if ($LASTEXITCODE -ne 0) {
    Write-Host "激活环境失败，请检查环境名称是否正确"
    exit 1
}

# 3. 依次运行Python脚本
Write-Host "`n正在运行第一个脚本: $pythonScript1"
python $pythonScript1
if ($LASTEXITCODE -ne 0) {
    Write-Host "$pythonScript1 执行失败"
    exit 1
}

Write-Host "`n正在运行第二个脚本: $pythonScript2"
python $pythonScript2
if ($LASTEXITCODE -ne 0) {
    Write-Host "$pythonScript2 执行失败"
    exit 1
}

Write-Host "`n正在运行第三个脚本: $pythonScript3"
python $pythonScript3
if ($LASTEXITCODE -ne 0) {
    Write-Host "$pythonScript3 执行失败"
    exit 1
}

Write-Host "`n所有脚本执行完成"