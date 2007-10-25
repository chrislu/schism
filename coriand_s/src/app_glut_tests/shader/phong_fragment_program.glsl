
#version 120

varying vec3 normal;
varying vec3 view_dir;

void main() 
{
    vec4 res;
    vec3 n = normalize(normal); 
    vec3 l = normalize(gl_LightSource[0].position.xyz); // assume parallel light!
    vec3 v = normalize(view_dir);
    vec3 h = normalize(l + v);//gl_LightSource[0].halfVector);//

    res =  gl_FrontLightProduct[0].ambient
         + gl_FrontLightProduct[0].diffuse * max(0.0, dot(n, l))
         + gl_FrontLightProduct[0].specular * pow(max(0.0, dot(n, h)), gl_FrontMaterial.shininess);
         
    gl_FragColor = res;
}

