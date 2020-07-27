#!/usr/bin/env bash
set -euo pipefail

binary_name=$1
build_dir="build_AppImage_${binary_name}"
app_dir=${build_dir}/AppDir

mkdir -p "$build_dir"
pushd $build_dir
cmake ..
make $binary_name
popd

realbin=${build_dir}/examples/${binary_name}

echo "[Desktop Entry]
Categories=Science;
Type=Application
Name=TinyDiffSim_$binary_name
Icon=app
Exec=$binary_name" > $build_dir/app.desktop
touch $build_dir/app.png

./tools/linuxdeploy-x86_64.AppImage \
  --appimage-extract-and-run \
  -e $realbin \
  -d$build_dir/app.desktop \
  -iapp.png \
  --appdir=$app_dir
./tools/appimagetool-x86_64.AppImage --appimage-extract-and-run $app_dir
