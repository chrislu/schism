
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

namespace scm {

inline bool is_host_little_endian()
{
    // based on wikipedia example code ;)
    int i = 1;
    char *p = (char *)&i;

    if (p[0] == 1) { // lowest address contains the lsb
        return (true);
    }
    else {
        return (false);
    }
}

template<typename T>
inline void swap_endian(T& val)
{
    unsigned int Tsize = sizeof(T);
    
    for (unsigned i = 0; i < Tsize/2; ++i) {
        unsigned char* r = (unsigned char*)(&val);
        unsigned char t;

        t                = r[i];
        r[i]             = r[Tsize - 1 - i];
        r[Tsize - 1 - i] = t;
    }
}

} // namespace scm
