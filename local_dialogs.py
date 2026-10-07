"""User-triggered local file selection; selected files remain data, never code."""
from __future__ import annotations
import os
import subprocess
import threading

_dialog_lock = threading.Lock()
_WINDOWS_PICKER = r'''
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$picker = New-Object System.Windows.Forms.OpenFileDialog
$picker.Title = 'Choose goSPL output: gospl.xdmf or an epoch XMF'
$picker.Filter = 'goSPL descriptors (*.xdmf;*.xmf)|*.xdmf;*.xmf|HDF5 results (*.h5;*.hdf5)|*.h5;*.hdf5'
$picker.CheckFileExists = $true
try {
    if ($picker.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Write($picker.FileName)
    }
} finally { $picker.Dispose() }
'''


def choose_gospl_result():
    if not _dialog_lock.acquire(blocking=False):
        raise ValueError('A file picker is already open.')
    try:
        if os.name != 'nt':
            raise ValueError('Paste the local goSPL output folder or descriptor path on this computer.')
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-STA', '-WindowStyle', 'Hidden', '-Command', _WINDOWS_PICKER],
            capture_output=True, encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=300)
        if result.returncode:
            raise ValueError('The file picker could not open. Paste the output folder or descriptor path instead.')
        return result.stdout.strip() or None
    except subprocess.TimeoutExpired:
        raise ValueError('The file picker timed out. Paste the output path or open it again.') from None
    finally:
        _dialog_lock.release()
