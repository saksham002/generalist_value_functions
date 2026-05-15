# Install uv

mkdir -p /nfs/aidm_nfs/jeffyu/uv/{bin,cache}
curl -LsSf https://astral.sh/uv/install.sh | \
  UV_INSTALL_DIR=/nfs/aidm_nfs/jeffyu/uv/bin \
  sh

echo '
# ---------- uv ----------
export UV_ROOT="/nfs/aidm_nfs/jeffyu/uv"
export PATH="$UV_ROOT/bin:$PATH"

# uv storage (no $HOME usage)
export UV_CACHE_DIR="$UV_ROOT/cache"
export UV_PROJECT_ENVIRONMENT="$UV_ROOT/vla"
# -----------------------------------------------
' >> ~/.bashrc

export UV_ROOT="/nfs/aidm_nfs/jeffyu/uv"
export PATH="$UV_ROOT/bin:$PATH"
export UV_CACHE_DIR="$UV_ROOT/cache"
export UV_PROJECT_ENVIRONMENT="$UV_ROOT/vla"
uv --version

# Install nasm

export NASM_PREFIX=/nfs/aidm_nfs/jeffyu/nasm
mkdir -p "$NASM_PREFIX/src"

cd "$NASM_PREFIX/src"

wget https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.gz
tar -xzf nasm-2.16.03.tar.gz
cd nasm-2.16.03

./configure --prefix="$NASM_PREFIX"
make -j$(nproc)
make install

cat >> ~/.bashrc <<'EOF'
# ---------- NASM ----------
export NASM_PREFIX="/nfs/aidm_nfs/jeffyu/nasm"
export PATH="$NASM_PREFIX/bin:$PATH"
# -----------------------------------------------
EOF

export PATH="/nfs/aidm_nfs/jeffyu/nasm/bin:$PATH"

which nasm
nasm -v

# Install pkg-config

export PKGCONFIG_PREFIX=/nfs/aidm_nfs/jeffyu/pkg-config
mkdir -p "$PKGCONFIG_PREFIX/src"
cd "$PKGCONFIG_PREFIX/src"

wget https://pkg-config.freedesktop.org/releases/pkg-config-0.29.2.tar.gz
tar -xzf pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2

./configure \
  --prefix="$PKGCONFIG_PREFIX" \
  --with-internal-glib

make -j"$(nproc)"
make install

export PATH="$PKGCONFIG_PREFIX/bin:$PATH"
which pkg-config
pkg-config --version

echo '
# ---------- pkg-config ----------
export PKGCONFIG_PREFIX="/nfs/aidm_nfs/jeffyu/pkg-config"
export PATH="$PKGCONFIG_PREFIX/bin:$PATH"
# -----------------------------------------------------
' >> ~/.bashrc

# Install ffmpeg-7

export FFMPEG_PREFIX=/nfs/aidm_nfs/jeffyu/ffmpeg-7
mkdir -p "$FFMPEG_PREFIX"/{src,bin,lib,include}

cd "$FFMPEG_PREFIX/src"

wget https://ffmpeg.org/releases/ffmpeg-7.0.2.tar.xz
tar -xf ffmpeg-7.0.2.tar.xz
cd ffmpeg-7.0.2
./configure \
  --prefix="$FFMPEG_PREFIX" \
  --enable-shared \
  --disable-static \
  --disable-doc \
  --disable-debug \
  --enable-pic

make -j$(nproc)
make install

# Checks
ls $FFMPEG_PREFIX/lib/libavformat.so*
ls $FFMPEG_PREFIX/lib/libavcodec.so*
ls $FFMPEG_PREFIX/include/libavformat/avformat.h
ldd $FFMPEG_PREFIX/bin/ffmpeg | grep libavformat
pkg-config --modversion libavformat
pkg-config --cflags libavformat
pkg-config --libs libavformat

cat >> ~/.bashrc <<'EOF'
# ---------- FFmpeg 7.x (local, no $HOME usage) ----------
export FFMPEG_PREFIX="/nfs/aidm_nfs/jeffyu/ffmpeg-7"
export PATH="$FFMPEG_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$FFMPEG_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# ------------------------------------------------------
EOF

export PATH="/nfs/aidm_nfs/jeffyu/ffmpeg-7/bin:$PATH"
export LD_LIBRARY_PATH="/nfs/aidm_nfs/jeffyu/ffmpeg-7/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="/nfs/aidm_nfs/jeffyu/ffmpeg-7/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
which ffmpeg
ffmpeg -version


# Setup uv environment

cd /nfs/aidm_nfs/jeffyu/batch_value_learning
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds
GIT_LFS_SKIP_SMUDGE=1 VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install -e .
VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install \
  --force-reinstall \
  "tensorflow-probability==0.23.0"

VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install \
  --force-reinstall \
  "jax[tpu]==0.5.3" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps "wandb==0.24.0"

VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install \
  --force-reinstall \
  "numpy==1.26.4"
