# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import sys
sys.setrecursionlimit(5000)

a = Analysis(['ADLERGUI.py'],
             pathex=['D:\\Profile\\jlb\\small_projects\\eric\\ADLER'],
             # binaries=[('mkl_intel_thread.dll','.')],
             binaries=[],
             datas=[('D:\\Profile\\jlb\\code\\ADLER\\adler2.ico', '.')],
             hiddenimports=['scipy.optimize'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='ADLER',
          debug=False,
          icon='.\\adler2.ico',
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
