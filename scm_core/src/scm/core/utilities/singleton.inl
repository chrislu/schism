
#include <iostream>
#include <cassert>
#include <stdexcept>

namespace scm {
} // namespace scm

#define SCM_SINGLETON_PLACEMENT(SCM_MODULE, SCM_CLASS)                                                  \
                                                                                                        \
namespace scm {                                                                                         \
                                                                                                        \
template<>                                                                                              \
boost::mutex singleton<SCM_CLASS>::_lock;                                                               \
                                                                                                        \
template<>                                                                                              \
typename singleton<SCM_CLASS>::instance_ptr_type singleton<SCM_CLASS>::_single_instance = 0;            \
                                                                                                        \
template<>                                                                                              \
bool singleton<SCM_CLASS>::_dead = false;                                                               \
                                                                                                        \
template<>                                                                                              \
__scm_export(SCM_MODULE) SCM_CLASS&                                                                     \
singleton<SCM_CLASS>::get()                                                                             \
{                                                                                                       \
    if (_single_instance == 0) { /* race condition */                                                   \
        create();/*system("pause");*/                                                                   \
    }                                                                                                   \
                                                                                                        \
    return (*_single_instance);                                                                         \
}                                                                                                       \
                                                                                                        \
template<>                                                                                              \
__scm_export(SCM_MODULE) void                                                                           \
singleton<SCM_CLASS>::create()                                                                          \
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
        _single_instance = new SCM_CLASS();                                                             \
    }                                                                                                   \
                                                                                                        \
    /* schedule destruction */                                                                          \
    std::atexit(&destroy);                                                                              \
}                                                                                                       \
                                                                                                        \
template<>                                                                                              \
__scm_export(SCM_MODULE) void                                                                           \
singleton<SCM_CLASS>::destroy()                                                                         \
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

