param([switch]$NoPause)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    if ($env:OS -ne "Windows_NT") { throw "AirDesk yêu cầu Windows 10/11 x64." }
    $py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $py)) {
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        $fallback = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($launcher) {
            & $launcher.Source -3.13 -m venv .venv
        } elseif ($fallback) {
            & $fallback.Source -c "import sys,struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8, 'Can Python 3.13 x64'"
            if ($LASTEXITCODE -ne 0) { throw "Hãy cài Python 3.13 x64 từ python.org, kèm Python Launcher." }
            & $fallback.Source -m venv .venv
        } else {
            throw "Không tìm thấy Python. Hãy cài Python 3.13 x64 từ python.org, chọn Python Launcher và Add python.exe to PATH, rồi mở lại tệp này."
        }
        if ($LASTEXITCODE -ne 0) { throw "Không tạo được .venv. Kiểm tra Python 3.13 x64 bằng: py -3.13 --version" }
    }
    & $py -c "import sys,struct; assert sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8, 'Can Python 3.13 x64'"
    if ($LASTEXITCODE -ne 0) { throw ".venv không dùng Python 3.13 x64. Đổi tên .venv để giữ bản cũ rồi chạy lại cài đặt." }
    Write-Host "Đang cài thư viện từ Internet. Lần đầu có thể mất vài phút..."
    & $py -m pip install "pip==26.2.1"
    if ($LASTEXITCODE -ne 0) { throw "Không cập nhật được pip. Kiểm tra kết nối Internet, proxy và chứng chỉ hệ thống." }
    & $py -m pip install --group runtime
    if ($LASTEXITCODE -ne 0) { throw "Cài thư viện thất bại. Xem lỗi pip ở trên; không tải DLL từ các trang không rõ nguồn gốc." }
    & $py -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Các phiên bản thư viện xung đột. Giữ bản .venv cũ bằng cách đổi tên và chạy lại cài đặt." }
    & $py scripts\check_install.py
    if ($LASTEXITCODE -ne 0) { throw "Kiểm tra thư viện hoặc tài nguyên thất bại. Xem thông báo ở trên." }
    Write-Host "Cài đặt AirDesk thành công. Chạy: .\Start AirDesk.bat"
    $result = 0
} catch {
    Write-Host ("LỖI: " + $_.Exception.Message) -ForegroundColor Red
    $result = 1
}
if (-not $NoPause) { Read-Host "Nhấn Enter để đóng cửa sổ" | Out-Null }
exit $result
