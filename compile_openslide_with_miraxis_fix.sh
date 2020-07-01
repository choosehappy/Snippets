apt update 
apt install -y  libtiff-dev libxml2-dev libopenjp2-7-dev  libcairo2-dev libcairo2  libgdk-pixbuf2.0-dev  sqlite3 libsqlite3-dev  git wget nano build-essential autoconf automake libtool pkg-config  python3-pip
cd / 
cd opt 
git clone https://github.com/openslide/openslide.git 

cd openslide/ 
autoreconf -i 

./configure



#-- apply patch from https://github.com/openslide/openslide/pull/293/commits/c907705f09f5ea5a3d8d0345768838409e2e5d19
 nano ./src/openslide-vendor-mirax.c
 
 
 change line 514- >  if (read_le_int32_from_file(f) < 1) {

#----

make -j 20 
make install 
 
 
pip3 install openslide-python numpy opencv-python


 
 
 
 
 
 
 
 
 
 

 
 
