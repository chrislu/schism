
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_OUT_STREAM_MANIP_H_INCLUDED
#define SCM_CORE_LOG_OUT_STREAM_MANIP_H_INCLUDED

#include <scm/core/log/out_stream.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace log {

__scm_export(core) out_stream& trace(out_stream& os);
__scm_export(core) out_stream& debug(out_stream& os);
__scm_export(core) out_stream& info(out_stream& os);
__scm_export(core) out_stream& output(out_stream& os);
__scm_export(core) out_stream& warning(out_stream& os);
__scm_export(core) out_stream& error(out_stream& os);
__scm_export(core) out_stream& fatal(out_stream& os);
__scm_export(core) out_stream& nline(out_stream& os);
__scm_export(core) out_stream& end(out_stream& os);
__scm_export(core) out_stream& flush(out_stream& os);
__scm_export(core) out_stream& indent(out_stream& os);
__scm_export(core) out_stream& outdent(out_stream& os);

class __scm_export(core) base_out_stream_manip
{
public:
    base_out_stream_manip() {}
    virtual ~base_out_stream_manip() {}
    virtual out_stream& do_manip(out_stream& los) const = 0;

}; // class base_out_stream_manip

class __scm_export(core) indent_fill : public base_out_stream_manip
{
public:
    indent_fill(out_stream::char_type c);
    virtual ~indent_fill();
    virtual out_stream& do_manip(out_stream& los) const;

    friend out_stream& operator<<(out_stream& os, const indent_fill& i) {
        return (i.do_manip(os));
    }

private:
    out_stream::char_type _c;
}; // class indent_fill

class __scm_export(core) indent_width : public base_out_stream_manip
{
public:
    indent_width(int w);
    virtual ~indent_width();
    virtual out_stream& do_manip(out_stream& los) const;

    friend out_stream& operator<<(out_stream& os, const indent_width& i) {
        return (i.do_manip(os));
    }

private:
    int _w;
}; // class indent_width

} // namespace log
} // namespace scm

#endif // SCM_CORE_LOG_OUT_STREAM_MANIP_H_INCLUDED
