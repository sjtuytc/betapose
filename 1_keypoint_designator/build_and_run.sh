cd build
rm -r *
cmake ../
make
./pcl-sift ../assets/models/obj_01.ply ../assets/sifts/sift_obj01.ply
# sz ../../assets/sifts/sift_obj01.ply