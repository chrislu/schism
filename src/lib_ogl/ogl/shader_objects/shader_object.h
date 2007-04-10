
#ifndef SHADER_OBJECT_H_INCLUDED
#define SHADER_OBJECT_H_INCLUDED

#include <deque>
#include <string>

namespace gl
{
    class program_object;

    class shader_object
    {
    public:
        explicit shader_object(unsigned int);
        virtual ~shader_object();

        void                    add_defines(const std::string& /*def*/);
        void                    add_include_code(const std::string& /*inc*/);
        bool                    add_include_code_from_file(const std::string& /*filename*/);
        void                    set_source_code(const std::string& /*src*/);
        bool                    set_source_code_from_file(const std::string& /*filename*/);

        bool                    compile();
        const std::string&      get_compiler_output() const { return (_compiler_out); }

    protected:

    private:
        bool                    get_source_from_file(const std::string& /*filename*/, std::string& /*out_code*/);

        unsigned int            _obj;
        unsigned int            _type;

        // source strings
        std::string             _inc;
        std::string             _def;
        std::string             _src;

        std::string             _compiler_out;

        friend class gl::program_object;
    };

} // namespace gl

#endif // SHADER_OBJECT_H_INCLUDED



