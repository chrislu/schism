
#ifndef VOLUME_TEXTURED_UNIT_CUBE_H_INCLUDED
#define VOLUME_TEXTURED_UNIT_CUBE_H_INCLUDED

namespace gl
{
    // this class represents a unit cube which extends exactly
    // one unit into x, y, z directions from the origin.
    //
    // texture coordinates are mapped to the vertices equal to
    // their positions (means v(0,0,0) has t(0,0,0) and v(1,1,1)
    // has t(1,1,1) and so on)
    class volume_textured_unit_cube
    {
    public:
        volume_textured_unit_cube();
        virtual ~volume_textured_unit_cube();

        bool                initialize();
        void                render(int) const;

    protected:
    private:
        void                clean_up();

        unsigned int        _vertices_vbo;
        unsigned int        _indices_vbo;
    };

} // namespace gl

#endif // VOLUME_TEXTURED_UNIT_CUBE_H_INCLUDED



