@echo off
REM Starts a local Qdrant server without Docker. Dashboard: http://127.0.0.1:6333/dashboard
REM Download qdrant-x86_64-pc-windows-msvc.zip from https://github.com/qdrant/qdrant/releases (v1.19.1)
REM and either unzip it to C:\precision-rag\qdrant_bin or set QDRANT_BIN to your qdrant.exe.
REM Keep storage OUTSIDE OneDrive/Dropbox folders (500k passages = ~1.8 GB).
if not defined QDRANT_BIN set QDRANT_BIN=C:\precision-rag\qdrant_bin\qdrant.exe
if not defined QDRANT_STORAGE set QDRANT_STORAGE=C:\precision-rag\qdrant_storage
set QDRANT__STORAGE__STORAGE_PATH=%QDRANT_STORAGE%
set QDRANT__STORAGE__SNAPSHOTS_PATH=%QDRANT_STORAGE%\snapshots
set QDRANT__TELEMETRY_DISABLED=true
"%QDRANT_BIN%"
