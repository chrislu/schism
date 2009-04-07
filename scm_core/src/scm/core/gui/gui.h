
#ifndef SCM_GUI_GUI_H_INCLUDED
#define SCM_GUI_GUI_H_INCLUDED

namespace scm {
namespace gui {

typedef enum
{
    orient_horizontal,
    orient_vertival
} text_orientation;

typedef enum
{
    hor_align_left,
    hor_align_right,
    hor_align_center
} text_hor_alignment;

typedef enum
{
    vert_align_bottom,
    vert_align_top,
    vert_align_center
} text_vert_alignment;

typedef enum
{
    flow_top_to_bottom,
    flow_bottom_to_top
} text_flow;

} // namespace gui
} // namespace scm

#endif // SCM_GUI_GUI_H_INCLUDED
