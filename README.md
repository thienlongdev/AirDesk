# AirDesk — Your Hands. Your Computer.

Điều khiển chuột và bàn phím Windows bằng cử chỉ bàn tay trước webcam. **Phiên bản chức năng: 1.1.4.** AirDesk dùng MediaPipe để suy luận các điểm mốc bàn tay, sau đó áp dụng hình học và máy trạng thái để quyết định thao tác; không huấn luyện mô hình AI riêng.

Kho mã nguồn: [thienlongdev/AirDesk](https://github.com/thienlongdev/AirDesk).

## 1. Giới thiệu và tính năng

AirDesk giúp thao tác máy tính khi muốn hạn chế chạm chuột/bàn phím, đồng thời là dự án thực hành thị giác máy tính, xử lý tín hiệu và lập trình giao diện Windows. Hệ thống chưa phải thiết bị trợ năng được chứng nhận hoặc công cụ thay thế chuột trong mọi tình huống.

| Chức năng | Cách hoạt động |
|---|---|
| Con trỏ năm ngón | Chọn tự động, thủ công hoặc khóa ngón; ánh xạ trực tiếp, tương đối và chính xác |
| Instant Auto-Click | Pinch được xác nhận gửi ngay một click hoàn chỉnh; giữ không tự lặp click |
| Smart Aim Lock | Giữ điểm ngắm khi các ngón khép để giảm lệch mục tiêu; tự nhả khi hủy pinch |
| Pinky Drag | Giơ riêng ngón út, thu ngón cái và các ngón còn lại, giữ **0,2 giây** để kéo |
| Cuộn chữ V | Giơ trỏ và giữa, thu áp út và út; dịch tay để cuộn dọc/ngang |
| Dual-Hand Air Keyboard | Hai con trỏ tay độc lập trên bàn phím: trái xanh dương, phải xanh bạc hà |
| Modifier một lần | Ctrl/Shift/Alt áp dụng cho phím kế tiếp từ bất kỳ tay nào; Ctrl+C/V/A/Z |
| Tiếng Việt | Bộ gõ Telex, xuất Unicode, gợi ý từ cục bộ tiếng Việt/Anh |
| Hiệu chỉnh | Webcam, tay/ngón, vùng thao tác, lọc rung, giao diện; lưu thiết lập/hồ sơ |
| Kiểm tra và an toàn | Diagnostics, Benchmark Lab; khởi động DISARMED, dừng khẩn cấp, nhả input khi mất tay/camera |

## 2. Công nghệ và yêu cầu hệ thống

| Thành phần | Vai trò / phiên bản cố định |
|---|---|
| Python **3.13 x64** | Môi trường nguồn đã kiểm chứng; máy kiểm thử dùng 3.13.9 |
| MediaPipe **1.1.1** | Hand Landmarker tối đa hai tay, CPU trên Windows |
| OpenCV contrib **5.0.0.93** | Webcam, đổi kích thước và màu ảnh |
| PySide6 **6.12.0** | Qt GUI, khay hệ thống, bàn phím không chiếm focus |
| NumPy **2.5.4** | Hình học và xử lý tín hiệu |
| Matplotlib **3.11.2**, Pillow **13.0.0** | Biểu đồ và xử lý ảnh phụ trợ |
| Win32 qua `ctypes` | `SendInput`, màn hình, focus, hotkey; không cần `pyautogui` |
| pytest / PyInstaller | Tùy chọn kiểm thử / đóng gói EXE |

- Windows 10/11 **64 bit**, Python 3.13 x64, webcam hoạt động trong ứng dụng Camera. Quy trình hiện được kiểm thử trên Windows 11; chưa thử trực tiếp trên Windows 10.
- Nên dùng webcam 720p khoảng 30 FPS, ánh sáng đều và thấy đủ bàn tay. Đây là khuyến nghị, không phải FPS tối thiểu được bảo đảm.
- CPU hiện đại, RAM đủ chạy Python/Qt/MediaPipe; chưa đo ngưỡng CPU/RAM tối thiểu. Không yêu cầu NVIDIA/CUDA.
- Internet để tải Python/thư viện lần đầu; nên để trống vài GB cho môi trường. Chạy ứng dụng thường không cần quyền quản trị.
- Nếu lỗi chỉ rõ thiếu runtime C++, cài [Microsoft Visual C++ Redistributable x64 chính thức](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist), không tải DLL từ nguồn lạ.

Các phiên bản lấy từ môi trường v1.1.4 và khai báo tại `pyproject.toml`. Những nhánh Python khác chưa được xác nhận cho quy trình này.

## 3. Tải mã nguồn

### Download ZIP

1. Mở [trang AirDesk](https://github.com/thienlongdev/AirDesk), chọn nhánh `main`.
2. Chọn **Code → Download ZIP**.
3. Nhấp phải ZIP → **Extract All… / Giải nén tất cả**. Không chạy bên trong cửa sổ ZIP.
4. Mở thư mục chứa `README.md`, `Setup AirDesk.bat` và `Start AirDesk.bat`.

[Tải ZIP nhánh main](https://github.com/thienlongdev/AirDesk/archive/refs/heads/main.zip). ZIP nguồn không có `.venv` hoặc EXE: cần thực hiện cài đặt dưới đây.

### Git clone (tùy chọn)

```powershell
git clone https://github.com/thienlongdev/AirDesk.git
cd AirDesk
```

Không cần Git nếu dùng ZIP. Đặt thư mục ở nơi có quyền ghi, ví dụ Documents; không cần một đường dẫn cố định trên ổ C.

## 4. Cài đặt chi tiết cho Windows

### Bước 1 — Python

Tải **Python 3.13 cho Windows x64** từ [python.org](https://www.python.org/downloads/windows/). Chọn **Python Launcher** và **Add python.exe to PATH** nếu trình cài đặt có các mục này. Mở lại PowerShell rồi kiểm tra:

```powershell
py -3.13 --version
py -3.13 -c "import struct; print(struct.calcsize('P') * 8)"
```

Cần thấy `Python 3.13.x` và `64`. Nếu thiếu `py`, sửa cài đặt để thêm Launcher. Bộ cài AirDesk cũng thử `python.exe` trong PATH, nhưng chỉ chấp nhận 3.13 x64.

### Bước 2 — Cài AirDesk

Nhấp đúp **Setup AirDesk.bat**, hoặc mở PowerShell **tại thư mục vừa giải nén**:

```powershell
& '.\Setup AirDesk.bat'
```

Bộ cài tạo `.venv` ngay trong dự án, kiểm tra Python, cài `pip==26.2.1`, cài nhóm `runtime` trong `pyproject.toml`, kiểm tra xung đột và nạp model. Thành công khi thấy **“Cài đặt AirDesk thành công”**. Có thể chạy lại khi tải bản nguồn mới.

Để tự động hóa: `& '.\Setup AirDesk.bat' -NoPause`.

Nếu Windows cảnh báo tệp tải xuống, kiểm tra nguồn và nội dung script trước khi cho phép. Script dùng `ExecutionPolicy Bypass` trong riêng tiến trình cài đặt, không đổi chính sách toàn hệ thống. Không cần tắt Defender hay chạy bằng quyền quản trị.

### Cài thủ công tương đương

Chạy tất cả lệnh tại thư mục chứa README:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==26.2.1
.\.venv\Scripts\python.exe -m pip install --group runtime
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\check_install.py
```

Không cần activate, `requirements.txt` hoặc cài editable. `pyproject.toml` dùng **dependency groups** của pip; chạy từ thư mục dự án, không dùng `pip install .`.

### Model và tài nguyên

`assets/models/hand_landmarker.task` được lưu trực tiếp trong Git, **không dùng Git LFS**. Dung lượng **7.819.105 byte ≈ 7,46 MiB**; SHA-256:

```text
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

Nguồn: [model Google chính thức, float16 phiên bản 1](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task); [tài liệu Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker). Bộ cài xác minh hash và nạp thử model, không âm thầm tải model thay thế. Nếu thiếu/hỏng, giải nén lại ZIP hoặc tải từ liên kết trên vào đúng đường dẫn rồi chạy `check_install.py`.

Sau khi cài đủ thư viện/model, suy luận ảnh chạy trên máy và không cần tải model mỗi lần. Theo [thông báo quyền riêng tư MediaPipe](https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe-tasks-privacy-notice), thư viện có thể gửi số liệu hiệu năng/sử dụng tới Google. “Suy luận cục bộ” không đồng nghĩa cam kết không có kết nối mạng của mọi thư viện.

## 5. Khởi chạy và cấu hình lần đầu

Nhấp đúp **Start AirDesk.bat**, hoặc:

```powershell
& '.\Start AirDesk.bat'
```

Lệnh dự phòng: `.\.venv\Scripts\python.exe -m airdesk`.

Launcher ưu tiên `.venv` của thư mục này. Khi không có môi trường nguồn nhưng có `dist\AirDesk\AirDesk.exe`, nó chạy EXE đó; thiếu cả hai thì yêu cầu Setup. Không tự dùng Python toàn cục. Khởi chạy thất bại trả mã thoát khác 0.

1. Ban đầu **DISARMED**, chưa điều khiển đầu vào.
2. Trong **Settings**, chọn webcam, backend, độ phân giải. Mặc định camera 0, MSMF, 1280×720, 30 FPS, soi gương. Diagnostics hiển thị chế độ thực nhận; webcam có thể không đáp ứng đúng yêu cầu.
3. Theo hướng dẫn lần đầu hoặc mở **Calibration** để hiệu chỉnh vùng điều khiển, khoảng cách và rung. Có thể bỏ qua rồi làm sau.
4. **Finger Control** chọn tay/ngón tự động hoặc thủ công. Kiểm tra preview và mirror trước khi ARM.
5. Nhấn **ARM GESTURE INPUT** hoặc `Ctrl+Alt+F11`. Thử trước trong Notepad.
6. `Ctrl+Alt+F12` dừng khẩn cấp. Đóng cửa sổ chính có thể chỉ thu xuống khay; chọn **Quit** ở khay để thoát hoàn toàn.

Thiết lập lưu tại `%APPDATA%\AirDesk\settings.json`, hồ sơ trong `profiles`, log xoay vòng trong `logs`, kết quả trong `bench_results`. `AIRDESK_HOME` đổi vị trí lưu khi thử nghiệm. Không chép dữ liệu cá nhân vào repo. AirDesk không chủ động ghi video webcam hoặc nội dung đã gõ vào log ứng dụng; bài kiểm tra có thể lưu văn bản mẫu do chính bài kiểm tra tạo ra trong thư mục tạm.

## 6. Hướng dẫn cử chỉ chuột

Đây là mặc định; có thể đổi tại **Gesture Mapping**.

| Cử chỉ | Thao tác |
|---|---|
| Chỉ một ngón, các ngón khác thu | Di chuyển; Auto chọn ngón đang chỉ. Hỗ trợ cả năm ngón |
| Pinch cái + ngón được gán | Instant click ngay sau xác nhận. Mở ngón giữa hai lần; hai pinch tách nhau tạo double click |
| Nhắm điểm rồi giơ **riêng út**, cái thu | Giữ **0,2 s** → nhấn giữ trái ở điểm đã ngắm; dịch cả tay để kéo; thu út để thả |
| V: trỏ + giữa duỗi, áp út + út thu | Cuộn dọc; dịch rõ theo ngang để cuộn ngang; bỏ V để dừng |
| 🤟: cái + trỏ + út duỗi, giữa + áp út thu | Giữ **0,4 s** hiện/ẩn bàn phím; thả tư thế trước lần sau |
| Nắm tay | Clutch: giữ con trỏ để đưa tay về chỗ thoải mái |
| Xòe tay vuốt phải / trái | Back / Forward nếu ứng dụng hỗ trợ |
| Hai tay pinch, tách/gần nhau | Zoom qua Ctrl+wheel trong chế độ chuột |
| Ngón cái làm con trỏ, ép vào cạnh trỏ | Thả trước 0,8 s → trái; 0,8 đến dưới 1,2 s → phải; giữ 1,2 s → kéo |

Ngón út có hai vai trò: Auto dùng út làm con trỏ khi **ngón cái đưa ra**. Út đơn độc và cái thu dành cho Pinky Drag. Chọn út trong Manual/Lock tắt Pinky Drag để tránh xung đột.

### Normal và Pinch Free

Đây là **kiểu ánh xạ nút**, khác chế độ click. Bảng cho **ngón trỏ làm con trỏ**; với ngón khác xem bảng tiếp xúc trên giao diện.

| Tiếp xúc | Normal | Pinch Free |
|---|---|---|
| Cái + trỏ | Trái | Không gán, trỏ tự do |
| Cái + giữa | Phải | Trái |
| Cái + áp út | Giữa | Phải |
| Cái + út | Không gán | Giữa |

### Chế độ click

- **Instant Auto-Click** mặc định: down+up khi xác nhận, giữ lâu không kéo/lặp. Dùng Pinky Drag để kéo.
- **Legacy Drag Mode**: giữ **1,2 s** mới kéo, thả sớm không click; mặc định quay về Instant sau một lần kéo.
- **Classic**: down khi xác nhận, up khi thả; giữ 1,2 s thành kéo. Click chỉ hoàn tất khi thả.

Smart Aim Lock dùng lịch sử điểm trỏ và tốc độ khép ngón để giữ điểm ngắm. Mở ngón/đổi ý sẽ giải phóng điểm giữ; đây là hình học và quy tắc thời gian, không phải AI đọc ý định.

| Phím vật lý mặc định | Chức năng |
|---|---|
| **Ctrl+Alt+F12** | Dừng khẩn cấp, nhả input |
| Ctrl+Alt+F11 | Bật/tắt ARM |
| Ctrl+Alt+F10 | Hiện/ẩn bàn phím, không cần nhận tay |
| Ctrl+Alt+F9 | Chế độ chính xác |
| Ctrl+Alt+F8 | Instant / Legacy Drag Mode |

Sửa tổ hợp trong Settings nếu bị trùng; Diagnostics báo đăng ký hotkey thất bại. Luôn giữ bàn phím/chuột vật lý trong tầm với.

## 7. Bàn phím hai tay, phím tắt và Telex

Mở bằng 🤟 hoặc `Ctrl+Alt+F10`. Mặc định Dual-Hand và lựa chọn `pinch`:

1. Tay trái **xanh dương**, phải **xanh bạc hà**, ký hiệu L/R. Đầu ngón vẽ trên overlay chính là điểm chọn phím.
2. Trỏ vào phím, pinch **tiếp xúc chính** của kiểu click đang chọn: Normal là cái+trỏ; Pinch Free là cái+giữa khi dùng trỏ. Mở contact để gõ tiếp.
3. Mỗi tay có lọc, pinch và aim riêng; hai phím cùng frame được nhận, một tay giữ không khóa tay kia. Backspace/Delete/mũi tên có cơ chế lặp khi giữ.
4. **Ctrl → C**, **Ctrl → V**, **Ctrl → A**, **Ctrl → Z** để sao chép/dán/chọn hết/hoàn tác. Modifier và ký tự có thể do hai tay khác nhau gõ. Ctrl/Shift/Alt dùng **một lần**, tự xóa sau phím kế tiếp; chạm lại để hủy. Trong cùng frame, modifier xử lý trước ký tự. Có các chord được phép như Ctrl+Shift+S.
5. Modifier chưa dùng hết hạn sau **10 s**, cả hai tay biến mất **1 s**, đóng bàn phím hoặc emergency stop.
6. Mất tay ngắn: nhớ track khoảng **0,3 s**; lâu hơn kết thúc track, dùng vị trí/nhãn để nhận lại. Tay trở lại một mình vẫn nhận được. Ghép tay mơ hồ tạm hoãn phím và hủy nếu mở pinch trước khi rõ identity; không bảo đảm đúng tuyệt đối khi tay chồng nhau.
7. Nút **EN/VI/RAW** đổi ngôn ngữ. **VI** dùng Telex: `Vieejt Nam` → `Việt Nam`; dấu `s/f/r/x/j`, chữ `aa/aw/ee/oo/ow/uw/dd`. Chọn **RAW** khi dùng IME hệ thống như UniKey, hoặc tắt IME ngoài khi dùng VI để tránh gõ dấu hai lần.
8. Thanh công cụ đổi layout, −/+, W−/W+, vị trí, gợi ý và ×; kích thước/vị trí được lưu. `dwell` và bàn phím một con trỏ vẫn có; dwell thuần không dùng Dual-Hand.

Overlay dùng `WS_EX_NOACTIVATE` để ứng dụng đích giữ focus. Chọn ô nhập trước rồi mở bàn phím. **Khi bàn phím hai tay mở, chuột hệ thống đứng yên và không click ngoài overlay**; đóng bằng 🤟/× để dùng chuột. Khi đổi ứng dụng hoặc click đổi caret, bộ ghép từ kết thúc để tránh sửa nhầm từ cũ.

Chặn từ bàn phím ảo: **Win+L, Ctrl+Esc, Alt+Tab, Alt+Esc, Ctrl+Shift+Esc**. Không nhận diện được mọi ô mật khẩu; bật **privacy mode** để tắt gợi ý và kết thúc từ khi cần.

## 8. Dừng khẩn cấp và quyền riêng tư

- Luôn khởi động DISARMED. ARM khi sẵn sàng và ứng dụng đích an toàn.
- **Ctrl+Alt+F12** độc lập webcam. Disarm, pause, lỗi xử lý và thoát đi qua đường nhả input.
- Mất tay quá grace sẽ nhả kéo. Camera không cấp frame hơn 150 ms làm runtime đưa frame rỗng vào engine để bảo vệ mất tracking vẫn chạy.
- Chuột vật lý được ưu tiên; sau khi dùng chuột thật, đợi khoảng 1,2 s để tay điều khiển lại.
- Không vượt UAC/UIPI/màn hình khóa. AirDesk thường không gửi được input vào cửa sổ có quyền cao hơn; thử ứng dụng không elevated.
- Stop Camera để nhường webcam, Quit để kết thúc các dịch vụ.
- Không đưa log, hồ sơ riêng, ảnh/video hoặc kết quả cá nhân lên GitHub. Model, biểu tượng và từ điển là tài nguyên cần thiết.

## 9. Cấu trúc thư mục

```text
AirDesk/
├── README.md                    # Hướng dẫn sử dụng và học mã nguồn
├── THIRD_PARTY_NOTICES           # Ghi công và giấy phép gốc
├── pyproject.toml               # Nhóm runtime / verify / build
├── Setup AirDesk.bat
├── Start AirDesk.bat
├── AirDesk.spec, launcher.py     # Đóng gói EXE
├── pytest.ini
├── airdesk/
│   ├── __main__.py, app.py       # Điểm vào Qt
│   ├── config.py, resources.py  # Cấu hình và tài nguyên
│   ├── runtime.py               # Dịch vụ, luồng, an toàn
│   ├── vision/                  # Webcam và MediaPipe
│   ├── core/                    # Hình học, cử chỉ, con trỏ, bàn phím, Telex
│   ├── platform/                # Win32, dispatcher, focus, hotkey
│   ├── ui/                      # GUI, hiệu chỉnh, overlay, Benchmark Lab
│   ├── data/                    # words_en.txt và words_vi.txt
│   ├── bench/                   # Thuật toán kiểm tra GUI/tests sử dụng
│   └── sim/                     # Tay mô phỏng cho tests/Benchmark Lab
├── assets/
│   ├── airdesk.ico, airdesk.png
│   └── models/                  # hand_landmarker.task, LICENSE Apache-2.0
├── scripts/
│   ├── setup.ps1, check_install.py, build.ps1
│   ├── check_click_reliability.py, check_mouse_pipeline.py
│   ├── check_keyboard_focus.py, check_v113.py, check_v114.py
│   └── _click_target.py, _editor_target.py, _keyboard_target.py, _windows_target.py
└── tests/                       # Hồi quy chức năng, an toàn, GUI
```

`bench`/`sim` được GUI/tests import; `_...target.py` là cửa sổ đích của kiểm tra tích hợp. `words_*.txt` được bộ gợi ý đọc khi chạy. `.venv`, `build`, `dist`, `release`, kết quả và tài liệu nội bộ cũ không công khai. README là tài liệu Markdown duy nhất.

## 10. Nguyên lý hoạt động

```text
Webcam → CameraManager → HandTracker / MediaPipe → FrameInput
       → InteractionEngine → cử chỉ / trạng thái / ánh xạ con trỏ
       → MouseDispatcher → Win32 SendInput → ứng dụng Windows
                         ↘ DualKeyboard → KeyboardOverlay (luồng Qt)
                                        → KeyboardOutput / Telex → SendInput
```

1. `python -m airdesk` vào `__main__.py` → `app.main()`: logging, Qt, khóa một phiên bản, đọc Settings, dựng Runtime và GUI.
2. `Runtime.start()` chạy dispatcher, hotkey, camera và vision. Engine chưa ARM.
3. `CameraManager._run()` chỉ giữ frame mới nhất; `wait_frame()` không tích hàng đợi ảnh cũ.
4. `Runtime._vision_loop()` gọi `HandTracker.process()`: thu nhỏ mặc định rộng 640 px, mirror, BGR→RGB, timestamp tăng đơn điệu → `detect_for_video()`.
5. MediaPipe trả tối đa hai tay, mỗi tay 21 landmark ảnh `(x,y,z)`, world landmark nếu có, nhãn và score. `HandObservation`/`FrameInput` đưa dữ liệu sang core.
6. `InteractionEngine.process()` chọn tay/ngón; `hand_model.analyze()` tính góc, độ duỗi, khoảng cách theo kích thước tay; `GestureRecognizer` áp dụng hysteresis và bằng chứng theo thời gian.
7. Máy trạng thái chọn POINTER, PRESS_PENDING, DRAGGING, SCROLLING, CLUTCH, KEYBOARD hoặc trạng thái an toàn. `PointerEngine` lọc và bù offset khi đổi ngón/nhả khóa.
8. `MouseSink` → `MouseDispatcher`: nút click gửi đồng bộ có kết quả; chuyển động làm mượt mặc định 120 Hz. **120 Hz không phải FPS camera/inference.**
9. Bàn phím mở: `DualKeyboard.update()` ghép quan sát với hai `VirtualHand`, dự đoán chuyển động và duy trì pinch/aim riêng. Queue đưa phím tới GUI; `KeyboardOutput.press()` xử lý modifier, Telex và Unicode/VK.
10. `win32.py` dựng `INPUT`, gọi `SendInput`. GUI đọc `Runtime.snapshot()` để cập nhật; không chạy inference trong luồng vẽ.
11. `Runtime.shutdown()` disarm, nhả chuột/phím, dừng dispatcher/vision/camera/hotkey, đóng tracker; `atexit` nhả thêm khi kết thúc thông thường.

## 11. Thứ tự đọc mã nguồn

Đọc đường chính theo thứ tự dưới; Benchmark Lab và chi tiết hiệu chỉnh có thể học sau.

| Tệp / thứ tự | Vai trò, thành phần và dữ liệu |
|---|---|
| 1. `airdesk/__main__.py`, `app.py` | `main`: tham số → Qt, Settings, Runtime, MainWindow; điểm bắt đầu/vòng đời |
| 2. `airdesk/config.py` | `Settings.load/save`, dataclass: JSON ↔ mặc định/di trú; app/runtime/UI dùng |
| 3. `airdesk/runtime.py` | `Runtime.start`, `_vision_loop`, `shutdown`: nối camera→tracker→engine, khóa và bảo vệ |
| 4. `airdesk/vision/camera.py` | `CameraManager`: cấu hình webcam → frame mới nhất + timestamp |
| 5. `airdesk/vision/tracker.py` | `HandTracker.process`: BGR → `FrameInput` + preview; lớp gọi model |
| 6. `airdesk/core/types.py` | `HandObservation`, `FrameInput`, `EngineStatus`, `Finger`: hợp đồng vision/core/UI |
| 7. `airdesk/core/hand_model.py` | `analyze`, `ily_pose`, `pinky_pose`: landmark → đặc trưng/tư thế |
| 8. `airdesk/core/selection.py` | `HandSelector`, `FingerSelector`: nhiều tay/ngón → lựa chọn ổn định |
| 9. `airdesk/core/gestures.py` | `GestureRecognizer`, `Hysteresis`, `slot_contacts`: đặc trưng/thời gian → contact/cử chỉ |
| 10. `airdesk/core/engine.py` | `InteractionEngine.process`, `_pinky_update`, `_start_pinky_drag`: frame/trạng thái → lệnh sink |
| 11. `airdesk/core/pointer.py`, `filters.py` | `PointerEngine`, `OneEuroFilter2D`, `AdaptiveKalman2D`: điểm tay → con trỏ ổn định/liên tục |
| 12. `airdesk/core/dual_keyboard.py` | `DualKeyboard.update`, `VirtualHand`: hai tay → hai cursor/phím, identity/mất tay/nhận lại |
| 13. `airdesk/core/actions.py`, `screen.py` | Sink/protocol và màn hình: mô tả thao tác không phụ thuộc Win32, test dùng sink ghi |
| 14. `airdesk/platform/dispatcher.py`, `win32.py` | `MouseDispatcher`, API Win32: lệnh → sự kiện Windows và kết quả gửi/nhả |
| 15. `airdesk/platform/keyboard_output.py` | `KeyboardOutput.press`, `ModState`, `Win32Injector`: phím/modifier → chord/Unicode |
| 16. `airdesk/core/telex.py`, `suggest.py`, `keyboard_layout.py` | `TelexComposer`, `Suggester`, `layout_rects`: phím → tiếng Việt/gợi ý; bố cục vẽ/hit-test |
| 17. `airdesk/ui/keyboard_overlay.py` | `KeyboardOverlay`: queue từ engine → GUI, con trỏ ảo, hit-test, gửi phím không lấy focus |
| 18. `airdesk/ui/main_window.py`, `pages.py` | `MainWindow`, `AppContext`, trang điều khiển: người dùng ↔ cấu hình/snapshot/khay |
| 19. `airdesk/platform/hotkeys.py`, `ui/calibration.py` | `HotkeyService`, `Wizard`: hotkey độc lập camera, hiệu chỉnh landmark → cấu hình |

### Thuật toán cần hiểu

- **Con trỏ:** pha đầu ngón với khớp, lọc One Euro/Kalman/EMA, ánh xạ vùng thao tác sang màn hình; offset trước/sau lọc giữ liên tục khi đổi ngón và kéo/thả.
- **Instant click:** khoảng cách chuẩn hóa, lịch sử đóng/mở, hysteresis, nhận pinch nhanh hoặc contact dừng gần ngưỡng. Một down/up, chỉ rearm sau mở contact; giữ nguyên ngưỡng v1.1.4.
- **Aim Lock:** giữ điểm trước lúc khép; hủy bởi mở ngón, dừng khép, thời gian tối đa hoặc di chuyển cả tay.
- **Pinky Drag:** tư thế riêng giữ 0,2 s; nhấn tại điểm ngắm ổn định trước đổi tư thế, kéo bằng tâm khớp, nhả khi tư thế hết/nhánh bảo vệ.
- **Cuộn:** V đưa vào SCROLLING; dịch tay thành lượng wheel, con trỏ giữ yên; có cuộn ngang.
- **Hai tay:** dự đoán cổ tay với vận tốc làm mượt, ghép có giới hạn khoảng cách; nhãn chỉ là một bằng chứng. Mỗi controller có vòng đời/pinch riêng.
- **Modifier:** batch chord modifier-down → key → modifier-up; đường lỗi nhả phần đã gửi; cùng frame ưu tiên modifier trước ký tự.

### Câu hỏi giảng viên có thể hỏi

| Câu hỏi | Trả lời ngắn |
|---|---|
| AI nằm ở đâu? | MediaPipe suy luận landmark; AirDesk dùng quy tắc chuyển thành thao tác |
| Có tự huấn luyện không? | Không, dùng model Google có nguồn/hash ở mục 4 |
| Bàn tay biểu diễn thế nào? | 21 điểm ảnh chuẩn hóa, world landmark nếu có, handedness và score |
| Vì sao thu nhỏ ảnh? | Giảm chi phí inference; landmark chuẩn hóa vẫn ánh xạ được lên màn hình |
| Nhận pinch thế nào? | Khoảng cách theo kích thước tay + độ duỗi + hysteresis + thời gian |
| Vì sao không chỉ xét một frame? | Nhiễu có thể tạo contact giả và click lặp |
| Click khác kéo thế nào? | Instant hoàn tất ngay; Pinky Drag tư thế riêng; legacy/classic giữ 1,2 s |
| Aim Lock giải quyết gì? | Khép ngón làm đầu ngón lệch dù người dùng muốn click điểm cũ |
| Vì sao lọc? | Landmark rung bị khuếch đại lên màn hình; phải cân bằng ổn định và trễ |
| Hai con trỏ có phải hai chuột Windows? | Không, là hai đối tượng ảo trên keyboard; Windows có một con trỏ |
| Tay mất rồi về một mình? | Vị trí nhớ và nhãn giúp tạo lại controller, không cần tay kia |
| Khi hai tay chồng nhau? | Hoãn kích hoạt liên quan, có thể bỏ phím để tránh gửi sai |
| Hai tay bị gõ tuần tự? | Nhận pinch độc lập; queue gửi input có thứ tự, modifier ưu tiên cùng frame |
| Gửi chữ Việt thế nào? | TelexComposer tạo Unicode, KeyboardOutput gửi phần thay đổi qua SendInput |
| Vì sao overlay giữ focus? | Kiểu NOACTIVATE và xử lý Qt để ứng dụng đích vẫn nhận phím |
| Vì sao nhiều luồng? | Camera/inference có thể chặn; GUI/hotkey không nên bị chặn theo |
| Camera mất khi kéo? | Frame rỗng đi qua nhánh mất tracking để nhả; shutdown nhả lại |
| Test mô phỏng chứng minh gì? | Hồi quy thuật toán/input; không chứng minh độ chính xác với mọi tay thật |
| Vì sao ánh sáng/FPS quan trọng? | Frame thưa, mờ/che khuất làm landmark và nhãn kém ổn định |
| Có hỗ trợ mọi ứng dụng không? | Không, UIPI/UAC/khóa màn hình/raw HID có thể chặn input |

## 12. Kiểm thử và đóng gói

### Kiểm tra không gửi input

```powershell
.\.venv\Scripts\python.exe scripts\check_install.py
.\.venv\Scripts\python.exe -m pip install --group verify
.\.venv\Scripts\python.exe -m pytest
```

Hồi quy bao phủ click trái/phải/giữa, Normal/Pinch Free, Aim Lock, Pinky Drag 0,2 s, V, 🤟 0,4 s, hai tay/nhận lại/gõ cùng frame, modifier, Telex, focus và nhả an toàn. Tay đầu vào là mô phỏng; giữ nguyên các assertion quan trọng.

### GUI và webcam DISARMED

```powershell
& '.\Start AirDesk.bat' --quit-after 15
```

Không ARM trong bài tự động này. `--quit-after` chỉ phục vụ kiểm thử. Xem Diagnostics để biết camera có frame/inference; mở được GUI không chứng minh mọi cử chỉ thật đúng.

### Tích hợp Windows — có input thật

Lưu công việc, đóng AirDesk đang chạy; các script tạo cửa sổ đích, di chuyển chuột/gõ văn bản mẫu. Không chạm chuột/phím hoặc chạy nhiều script đồng thời.

```powershell
.\.venv\Scripts\python.exe scripts\check_mouse_pipeline.py
.\.venv\Scripts\python.exe scripts\check_keyboard_focus.py
.\.venv\Scripts\python.exe scripts\check_v113.py
.\.venv\Scripts\python.exe scripts\check_v114.py
```

v113 kiểm tra click/Pinky Drag/phím tắt/Telex; v114 kiểm tra lifecycle hai tay, nhận lại một mình và modifier cùng frame. Chúng vẫn bổ sung nhau ở v1.1.4. `check_click_reliability.py` dùng khi chẩn đoán sâu click; giữ các helper mà script import. Kết quả JSON trong `bench_results` không được Git theo dõi.

`_windows_target.py` chỉ đưa cửa sổ thuộc chính tiến trình kiểm thử lên foreground khi chạy từ terminal nền. Nếu không lấy được focus, bài kiểm tra không được coi là đạt; helper này không tham gia hành vi focus của AirDesk.

Đợt xác minh nguồn ngày **19/10/2026** trên Windows 11 / Python 3.13.9 x64: cài từ ZIP xuất độc lập với môi trường ảo mới, `pip check` đạt, **343 tests đạt**, tích hợp v113 **26/26**, v114 **17/17**, kiểm tra chuột và focus/Unicode đạt. GUI đủ 10 trang, overlay và webcam 1280×720 chạy DISARMED, nhả input và đóng camera sạch. Các mốc này là kết quả trên máy kiểm thử, không phải cam kết độ chính xác với bàn tay thật.

### Xác nhận bằng tay thật

Benchmark Lab có bài thử: click ba nút ở hai kiểu, mục tiêu nhỏ, kéo/thả, cuộn dọc/ngang, năm lần 🤟, từng tay gõ riêng, mỗi tay rời camera rồi về **một mình**, hai pinch đồng thời, Ctrl+C/V chéo tay, Telex và emergency stop khi kéo. Ghi cả lần trượt/sai; tự động hóa không thay được bước này.

### EXE tùy chọn

```powershell
.\.venv\Scripts\python.exe -m pip install --group build
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1
```

Script chạy tests, PyInstaller theo `AirDesk.spec`, tạo `dist\AirDesk\AirDesk.exe` và `release\AirDesk-1.1.4-win64.zip`. ZIP **EXE** khác ZIP **nguồn GitHub**. Phân phối cả thư mục DLL/tài nguyên và giấy phép, không chỉ một EXE. Khởi động cùng Windows chỉ đăng ký từ bản EXE sau xác nhận trong Settings, vẫn DISARMED.

## 13. Khắc phục lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Thiếu `py` / Python | Cài 3.13 x64/Launcher, mở terminal mới; kiểm tra alias Microsoft Store nếu `python` trỏ nhầm |
| `.venv` sai Python hoặc đã chuyển chỗ | Đổi tên để giữ bản cũ rồi chạy Setup; không chép venv giữa máy/thư mục |
| pip lỗi mạng/chứng chỉ | Kiểm tra mạng/proxy/ngày giờ và lỗi pip; không bỏ TLS hoặc tải DLL lạ |
| Thiếu module / DLL load failed | Chạy Setup đúng `.venv`, kiểm tra x64 và runtime C++ nếu lỗi yêu cầu; chạy `pip check` |
| Camera bận/không mở | Thử ứng dụng Camera, bật quyền Camera cho desktop apps; đóng Zoom/Teams/ứng dụng camera khác |
| Camera sai / ảnh đen | Chọn lại index, quét thiết bị trong Settings; thử 720p/MSMF rồi DSHOW nếu không tương thích |
| Không nhận tay / quá gần/xa | Thấy đủ cổ tay/đầu ngón, sáng phía trước, tránh ngược sáng; hiệu chỉnh lại |
| Nhãn tay/màu sai | Kiểm tra mirror, lựa chọn tay; để hai tay tách rõ; xem Diagnostics hand identity |
| Có preview nhưng không điều khiển | Kiểm tra ARM/pause/nhường chuột vật lý và engine state; không đổi ngưỡng tùy tiện |
| Không click cửa sổ quản trị | UIPI chặn; dùng cửa sổ đích không elevated, không tìm cách vượt UAC |
| 🤟 không mở | Kiểm tra gesture bật, cái/trỏ/út và giữ 0,4 s; dùng Ctrl+Alt+F10 để tách lỗi gesture/overlay |
| Bàn phím gõ sai chỗ | Chọn ô nhập đích trước, tránh elevated; kiểm tra ngôn ngữ/modifier/cách chọn phím |
| Telex lặp dấu | VI + tắt IME ngoài, hoặc RAW + IME hệ thống |
| Model thiếu/hash sai | Tải lại ZIP/model chính thức; tệp khoảng 7,46 MiB, không phải LFS pointer; chạy `check_install.py` |
| Hotkey trùng | Xem Diagnostics, sửa tổ hợp trong Settings; dùng Disarm/khay |
| Đóng cửa sổ mà webcam còn bật | Còn chạy ở khay; chọn Quit hoặc Stop Camera |

## 14. Giới hạn và mức kiểm chứng

- Camera, thời gian phơi sáng và inference giới hạn tốc độ. Tối, nhòe, ngón khuất hoặc tay chồng nhau có thể gây bỏ sót/sai landmark.
- Handedness là dự đoán; tay mới có nhãn sai liên tục có thể mang màu tay kia. Lịch sử/dự đoán giảm lỗi, không phải định danh sinh trắc.
- Trỏ gập mạnh khi pinch phải có thể làm tiếp xúc cái+trỏ cạnh tranh với tiếp xúc khác. Không cam kết độ chính xác tuyệt đối.
- Dispatcher 120 Hz không tăng FPS camera. GPU trên Windows có thể fallback CPU. Chưa hỗ trợ đầy đủ ngoài Windows.
- Raw HID, game, remote desktop, UIPI, UAC và màn hình khóa có thể từ chối input giả lập.
- Test landmark mô phỏng, SendInput vào cửa sổ thử và smoke webcam là các bằng chứng khác nhau; độ tiện dụng với tay thật cần người dùng xác nhận.

Dọn nguồn giữ nguyên thuật toán/ngưỡng v1.1.4. Không coi số benchmark mô phỏng hoặc báo cáo cũ là cam kết hiệu năng trên máy mới.

## 15. Giấy phép và ghi công

Chủ dự án: **Nguyen Tu Thien Long**. Mã được triển khai với sự hỗ trợ của công cụ AI theo đặc tả chủ dự án; nguồn và phần thích nghi ở [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Không tuyên bố mọi thuật toán là phát minh mới.

**Chủ dự án chưa chọn giấy phép mã nguồn mở cho mã riêng AirDesk.** Kho công khai không tự cấp MIT/Apache cho phần đó. Khi cần quyền tái sử dụng/phân phối ngoài quyền sẵn có, liên hệ chủ dự án. Bên thứ ba vẫn theo giấy phép tương ứng:

- AirTouch (MIT): ý tưởng Kalman thích nghi; AirCursor (MIT): theo cùng cổ tay.
- Alpha-OSK (MIT): kinh nghiệm INPUT, Unicode và NOACTIVATE.
- Casiez, Roussel & Vogel, CHI 2012: One Euro Filter; không chép mã tham chiếu.
- MediaPipe/model Google: Apache-2.0; giữ [LICENSE của model](assets/models/LICENSE). Các thư viện pip kèm license/notice riêng.
- Qt for Python: LGPL-3.0/GPL/commercial theo thành phần; OpenCV Apache-2.0; NumPy/Matplotlib/Pillow theo giấy phép phân phối tương ứng.

Giữ nguyên văn giấy phép/notice. Các tệp này và `words_*.txt` là ngoại lệ cần thiết; không có tài liệu Markdown tách rời README.
