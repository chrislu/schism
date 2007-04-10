
#include <cassert>

namespace scm
{
    namespace core
    {
        template <class instance_t>
        template <class instance_t_>
        global_system_access<instance_t>::global_system_access_<instance_t_>::global_system_access_()
            : _instance(0)
        {
        }

        template <class instance_t>
        template <class instance_t_>
        global_system_access<instance_t>::global_system_access_<instance_t_>::~global_system_access_()
        {
        }

        template <class instance_t>
        template <class instance_t_>
        instance_t_& global_system_access<instance_t>::global_system_access_<instance_t_>::get() const
        {
            assert(_instance != 0);

            return (*_instance);
        }

        template <class instance_t>
        template <class instance_t_>
        instance_t_& global_system_access<instance_t>::global_system_access_<instance_t_>::operator*() const
        {
            assert(_instance != 0);

            return (*_instance);
        }

        template <class instance_t>
        template <class instance_t_>
        instance_t_*const global_system_access<instance_t>::global_system_access_<instance_t_>::get_ptr() const
        {
            assert(_instance != 0);

            return (_instance);
        }

        template <class instance_t>
        template <class instance_t_>
        instance_t_*const global_system_access<instance_t>::global_system_access_<instance_t_>::operator->() const
        {
            assert(_instance != 0);

            return (_instance);
        }

        template <class instance_t>
        template <class instance_t_>
        void global_system_access<instance_t>::global_system_access_<instance_t_>::set_instance(instance_t_*const inst) const
        {
            _instance = inst;
        }

    } // namespace core
} // namespace scm

