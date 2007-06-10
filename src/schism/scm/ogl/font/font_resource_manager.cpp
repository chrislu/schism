
#include "font_resource_manager.h"

using namespace scm::gl;

font_resource_manager::font_resource_manager()
{
}

font_resource_manager::~font_resource_manager()
{
}

bool font_resource_manager::initialize()
{
    return (true);
}

bool font_resource_manager::shutdown()
{
    bool ret = res::resource_manager<font_face_resource>::shutdown();

    return (ret);
}

