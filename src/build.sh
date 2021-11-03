# preprocess module
cd preproc
if [ -e ./preproc.py -a -e ./_preproc.so]; then
  rm preproc.py _preproc.so
fi
if [ -e ./build ]; then
  rm -r build
fi
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE .. && make
cp preproc.py _preproc.so ..
