' VBScript khởi động giao diện phần mềm chuyển đổi PDF sang Word ẩn cửa sổ dòng lệnh đen
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c Chay_Giao_Dien.bat", 0, False
