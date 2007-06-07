
#include <cassert>

namespace scm {
namespace core {

template <class instance_t>
template <class instance_t_>
core_system_singleton<instance_t>::core_system_singleton_<instance_t_>::core_system_singleton_()
    : _instance(0)
{
}

template <class instance_t>
template <class instance_t_>
core_system_singleton<instance_t>::core_system_singleton_<instance_t_>::~core_system_singleton_()
{
}

template <class instance_t>
template <class instance_t_>
instance_t_& core_system_singleton<instance_t>::core_system_singleton_<instance_t_>::get() const
{
    assert(_instance != 0);

    return (*_instance);
}

template <class instance_t>
template <class instance_t_>
instance_t_& core_system_singleton<instance_t>::core_system_singleton_<instance_t_>::operator*() const
{
    assert(_instance != 0);

    return (*_instance);
}

template <class instance_t>
template <class instance_t_>
instance_t_*const core_system_singleton<instance_t>::core_system_singleton_<instance_t_>::get_ptr() const
{
    assert(_instance != 0);

    return (_instance);
}

template <class instance_t>
template <class instance_t_>
instance_t_*const core_system_singleton<instance_t>::core_system_singleton_<instance_t_>::operator->() const
{
    assert(_instance != 0);

    return (_instance);
}

template <class instance_t>
template <class instance_t_>
void core_system_singleton<instance_t>::core_system_singleton_<instance_t_>::set_instance(instance_t_*const inst) const
{
    _instance = inst;
}

} // namespace core
} // namespace scm
