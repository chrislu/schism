
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <iostream>
#include <cassert>
#include <stdexcept>

namespace scm {
} // namespace scm

#define SCM_SINGLETON_PLACEMENT(SCM_MODULE, SCM_CLASS)                                                  \
                                                                                                        \
namespace scm {                                                                                         \
                                                                                                        \
template<class managed_type>                                                                            \
boost::mutex singleton<managed_type>::_lock;                                                            \
                                                                                                        \
template<class managed_type>                                                                            \
typename singleton<managed_type>::instance_ptr_type singleton<managed_type>::_single_instance = 0;      \
                                                                                                        \
template<class managed_type>                                                                            \
bool singleton<managed_type>::_dead = false;                                                            \
                                                                                                        \
template<class managed_type>                                                                            \
managed_type&                                                                                           \
singleton<managed_type>::get()                                                                          \
{                                                                                                       \
    if (_single_instance == 0) { /* race condition */                                                   \
        create();/*system("pause");*/                                                                   \
    }                                                                                                   \
                                                                                                        \
    return (*_single_instance);                                                                         \
}                                                                                                       \
                                                                                                        \
template<class managed_type>                                                                            \
void                                                                                                    \
singleton<managed_type>::create()                                                                       \
{                                                                                                       \
    boost::mutex::scoped_lock   lock(_lock);                                                            \
                                                                                                        \
    if (_dead) {                                                                                        \
        std::cerr << "trying to recreate allready dead singleton instance ("                            \
                  << #SCM_CLASS                                                                         \
                  << ")" << std::endl;                                                                  \
        throw std::logic_error("fatal error: trying to recreate allready dead singleton instance");     \
    }                                                                                                   \
    if (_single_instance == 0) { /* race condition resolved */                                          \
        _single_instance = new managed_type();                                                          \
    }                                                                                                   \
                                                                                                        \
    /* schedule destruction */                                                                          \
    std::atexit(&destroy);                                                                              \
}                                                                                                       \
                                                                                                        \
template<class managed_type>                                                                            \
void                                                                                                    \
singleton<managed_type>::destroy()                                                                      \
{                                                                                                       \
    boost::mutex::scoped_lock   lock(_lock);                                                            \
                                                                                                        \
    if (_single_instance) {                                                                             \
        delete _single_instance;                                                                        \
        _single_instance = 0;                                                                           \
        _dead            = true;                                                                        \
    }                                                                                                   \
    else {                                                                                              \
        std::cerr << "trying to delete dead singleton instance ("                                       \
                  << #SCM_CLASS                                                                         \
                  << ")" << std::endl;                                                                  \
        throw std::logic_error("fatal error: trying to delete dead singleton instance");                \
    }                                                                                                   \
}                                                                                                       \
                                                                                                        \
} /* namespace scm */                                                                                   \
                                                                                                        \
template class __scm_export(SCM_MODULE) scm::singleton<SCM_CLASS>;                                      \

