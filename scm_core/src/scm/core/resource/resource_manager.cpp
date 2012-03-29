
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#if 0

#include "resource_manager.h"

#include <scm/core/resource/resource_pointer.h>

namespace scm {
namespace res {

resource_manager_base::resource_manager_base()
{
}

resource_manager_base::~resource_manager_base()
{
    clear_resources();
}

bool resource_manager_base::shutdown()
{
    clear_resources();

    return (true);
}

bool resource_manager_base::is_loaded(const resource_pointer_base& inst) const
{
    resource_container::const_iterator    res_it = _resources.find(inst._resource.lock()->hash_value());

    if (res_it != _resources.end()) {
        return (true);
    }
    else {
        return (false); // we should never get here!
    }
}

bool resource_manager_base::is_loaded(const resource_base::hash_type hash) const
{
    resource_container::const_iterator    res_it = _resources.find(hash);

    if (res_it != _resources.end()) {
        return (true);
    }
    else {
        return (false);
    }
}
//
//resource_pointer_base resource_manager_base::retrieve_instance(const resource_base::hash_type hash)
//{
//    resource_container::const_iterator    res_it = _resources.find(hash);
//
//    if (res_it != _resources.end()) {
//        return(resource_pointer_base(res_it->second.first, shared_from_this()));
//    }
//    else {
//        return(resource_pointer_base());
//    }
//}

void resource_manager_base::register_instance(const resource_pointer_base& inst)
{
    resource_container::iterator    res_it = _resources.find(inst._resource.lock()->hash_value());

    if (res_it != _resources.end()) {
        res_it->second.second += 1;
    }
    else {
        assert(0); // we should never get here!
    }

    // the only instance is supposed to be in here!
    //assert(inst._resource.use_count() == 1);
}

void resource_manager_base::release_instance(const resource_pointer_base& inst)
{
    resource_container::iterator    res_it = _resources.find(inst._resource.lock()->hash_value());

    if (res_it != _resources.end()) {
        res_it->second.second -= 1;

        if (res_it->second.second < 1) {
            res_it->second.first.reset();

            _resources.erase(res_it);

            assert(inst._resource.use_count() == 0);
        }
    }
    else {
        assert(0); // we should never get here!
    }
}

void resource_manager_base::insert_instance_values(const resource_base::hash_type hash,
                                                   const res_ptr_type&            ptr)
{
    _resources.insert(resource_container::value_type(hash,
                                                     std::make_pair(ptr, 1)));
}

void resource_manager_base::clear_resources()
{
    resource_container::iterator    res_it;

    for (res_it =  _resources.begin();
         res_it != _resources.end();
         ++res_it) {

        res_it->second.first.reset();
    }

    _resources.clear();
}

} // namespace res
} // namespace scm

#endif // 0

