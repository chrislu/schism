
#ifndef SCM_GL_UTIL_VGEO_H_INCLUDED
#define SCM_GL_UTIL_VGEO_H_INCLUDED

// reverse engineered vgeo format
// - header size 3072 bytes
// - interesting stuff is the bpv and size
// - values stored big endian

struct vgeo_header {
    int             _magic;
    int             _volume_type;

    unsigned char   _unused_00[324];

    int             _bits_per_voxel;
    int             _size_x;
    int             _size_y;
    int             _size_z;

    unsigned char   _unused_01[3072 - 348];
};

#endif // SCM_GL_UTIL_VGEO_H_INCLUDED
