
#include <scm_core/resource/resource.h>

namespace scm {
namespace res {

template<class res_type>
resource_manager<res_type>::resource_manager()
{
    _this.reset(this);
}


template<class res_type>
resource_manager<res_type>::~resource_manager()
{
    clear_instances();
}

template<class res_type>
resource<res_type>
resource_manager<res_type>::find_instance(const typename resource_manager<res_type>::res_desc_type& desc)
{
    instance_container::iterator    known_inst = _instances.find(desc);

    if (known_inst != _instances.end()) {
        return (resource<res_type>(known_inst->second.first, _this));
    }
    else {
        return (resource<res_type>());
    }    
}

template<class res_type>
resource<res_type>
resource_manager<res_type>::create_instance(const typename resource_manager<res_type>::res_desc_type& desc)
{
    resource<res_type>  ret = find_instance(desc);

    if (!ret) {
        boost::shared_ptr<res_type>    new_res(new res_type(desc));   

        _instances.insert(instance_container::value_type(desc,
                                                         std::make_pair(new_res, 1)));
        ret = resource<res_type>(new_res, _this);
    }

    // the only instance is supposed to be in here!
    assert(ret._resource.use_count() == 1);

    return (ret);
}

template<class res_type>
void resource_manager<res_type>::register_instance(const resource<res_type>& inst)
{
    if (boost::shared_ptr<res_type> i = inst._resource.lock()) {
        instance_container::iterator    known_inst = _instances.find(i->get_descriptor());

        if (known_inst != _instances.end()) {
            known_inst->second.second += 1;
        }
        else {
            assert(0);
            //_instances.insert(instance_container::value_type(i->get_descriptor(),
            //                                                 std::make_pair(i, 1)));
        }
    }

    // the only instance is supposed to be in here!
    assert(inst._resource.use_count() == 1);
}

template<class res_type>
void resource_manager<res_type>::release_instance(const resource<res_type>& inst)
{
    if (boost::shared_ptr<res_type> i = inst._resource.lock()) {

        instance_container::iterator    known_inst = _instances.find(i->get_descriptor());

        if (known_inst != _instances.end()) {
            known_inst->second.second -= 1;

            if (known_inst->second.second < 1) {
                known_inst->second.first.reset();

                _instances.erase(known_inst);

                // the only instance is supposed to be the locked i
                assert(inst._resource.use_count() == 1);
            }
        }
        else {
            assert(0);
            // wtf this is impossible... impossible... THIS IS SPARTAAAAA
            // wtf where is this instance coming from?
        }
    }
}

template<class res_type>
void resource_manager<res_type>::clear_instances()
{
    instance_container::iterator    inst;

    for (inst =  _instances.begin();
         inst != _instances.end();
         ++inst) {

        inst->second.first.reset();
    }

    _instances.clear();
}

} // namespace res
} // namespace scm
