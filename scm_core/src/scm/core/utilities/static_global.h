
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_STATIC_GLOBAL_H_INCLUDED
#define SCM_CORE_STATIC_GLOBAL_H_INCLUDED

#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include <boost/thread/mutex.hpp>

/*
    usage:

    *.cpp
    SCM_STATIC_GLOBAL(std::vector<something>, _my_global_static);

    void some_func() {
        _my_global_static().push_back();
    }
*/

namespace scm {

template<typename managed_type>
class static_global
{
public:
    typedef managed_type* static_type;

    static_global() : _pointer(0), _alive(true) {}
    ~static_global()
    {
        if (_pointer != 0) {
            //std::cout << "deleting " << _pointer << std::endl;
            delete _pointer;
            _pointer = 0;
            _alive   = false;
        }
    }

    static_type     _pointer;
    bool            _alive;
}; // static_global

} // namespace scm

#define SCM_STATIC_GLOBAL(SCM_TYPE, SCM_NAME)                                                                           \
static SCM_TYPE& SCM_NAME()                                                                                             \
{                                                                                                                       \
    static scm::static_global<SCM_TYPE >    static_ptr;                                                                 \
    static boost::mutex                     static_init_mutex;                                                          \
                                                                                                                        \
    if (static_ptr._pointer == 0) { /* race */                                                                          \
        boost::mutex::scoped_lock   lock(static_init_mutex);                                                            \
                                                                                                                        \
        if (!static_ptr._alive) {                                                                                       \
            std::cerr << "trying to recreate allready dead global static instance ("                                    \
                      << #SCM_TYPE                                                                                      \
                      << " "                                                                                            \
                      << #SCM_NAME                                                                                      \
                      << ")" << std::endl;                                                                              \
            throw std::logic_error("fatal error: trying to recreate allready dead global static instance");             \
        }                                                                                                               \
        if (static_ptr._pointer == 0) { /* race resolved */                                                             \
            static_ptr._pointer = new SCM_TYPE();                                                                       \
        }                                                                                                               \
    }                                                                                                                   \
                                                                                                                        \
    return (*static_ptr._pointer);                                                                                      \
}

#endif // SCM_CORE_STATIC_GLOBAL_H_INCLUDED
