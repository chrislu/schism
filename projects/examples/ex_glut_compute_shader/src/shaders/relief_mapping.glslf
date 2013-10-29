
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

// uniform variables
uniform float shininess;
uniform float depth;
uniform float lowestLevel;
uniform float linear_step_size;
uniform float shadow_step_size;
uniform int binary_search_steps;
uniform float max_mip_level;

// eye space variables
varying vec3 eyeSpaceTangent;
varying vec3 eyeSpaceBinormal;
varying vec3 eyeSpaceNormal;

varying vec3 eyeSpaceVert;
varying vec3 eyeSpaceLight;


// textures
uniform sampler2D reliefMap;
uniform sampler2D quadTreeMap;
uniform sampler2D colorMap;
uniform sampler2D normalMap;
uniform sampler2D ambientMap;

struct Ray
{
    vec3    rayDir;
    vec3    invRayDir;
    vec3    intersectBboxMax;
};

bool compute_shadow(in vec3 intersectPoint, in vec3 eyeSpaceLightDir);
float get_nearest_border_2D(in vec2 startPoint, in Ray currentRay, in float nodesPerRow);
float compute_exitparam(in vec3 intersectPoint, in Ray currentRay);
float quadtree_search(in vec3 startPoint, in Ray currentRay, in float mip_level);
float linear_search(in vec3 startPoint, in Ray currentRay, in float step_size);
vec3 compute_normal(in vec2 texPoint);

void main(void)
{
   vec4 texColor, quadTreeData, projTexPoint;
   vec3 eyeSpaceEyeDir, eyeSpaceLightDir, eyeSpaceReflectDir, texPoint, normalData, finalColor, startPoint;
   float eyeRayParam;
   Ray currentRay;

   // normalize light and eye ray
   eyeSpaceEyeDir = normalize(eyeSpaceVert);

   // transform eye direction into tangent space
   eyeSpaceEyeDir = normalize(vec3(dot(eyeSpaceEyeDir,  eyeSpaceTangent),
	   					   		   dot(eyeSpaceEyeDir,  eyeSpaceBinormal),
   					   			   dot(-eyeSpaceEyeDir, eyeSpaceNormal)));

   currentRay.rayDir = eyeSpaceEyeDir;

   currentRay.invRayDir = vec3(1.0) / currentRay.rayDir;

   vec3 signRay = sign(currentRay.rayDir);
   currentRay.intersectBboxMax.x = (signRay.x > 0.0) ? currentRay.invRayDir.x : 0.0;
   currentRay.intersectBboxMax.y = (signRay.y > 0.0) ? currentRay.invRayDir.y : 0.0;
   currentRay.intersectBboxMax.z = (signRay.z > 0.0) ? currentRay.invRayDir.z : 0.0;

   startPoint = gl_TexCoord[0].xyz;

   startPoint.z = abs(depth - gl_TexCoord[0].z) < 0.0001 ? depth : gl_TexCoord[0].z;

   // ray intersect in view direction
   eyeRayParam = quadtree_search(startPoint, currentRay, max_mip_level - 1.0);
   texPoint = startPoint + eyeRayParam * currentRay.rayDir.xyz;

   eyeRayParam = linear_search(startPoint + eyeRayParam * currentRay.rayDir.xyz, currentRay, linear_step_size);
   texPoint = texPoint + eyeRayParam * currentRay.rayDir.xyz;

   texColor = texture2D(colorMap, texPoint.xy);

   // expand normal from normal map in local polygon space
   normalData = compute_normal(texPoint.xy);

   eyeSpaceLightDir = normalize(eyeSpaceLight - (gl_ModelViewMatrix * vec4(texPoint, 1.0)).xyz);
   eyeSpaceReflectDir = normalize(-reflect(eyeSpaceLightDir, normalData.xyz));

   // compute diffuse and specular terms
   float LDotN = dot(eyeSpaceLightDir ,normalData.xyz);
   float att = max(0.0, LDotN);
   float diff = max(0.0, dot(eyeSpaceLightDir, normalData.xyz));
   float spec = max(0.0, dot(eyeSpaceEyeDir, eyeSpaceReflectDir));

   quadTreeData = texture2D(reliefMap, startPoint.xy);

   // compute final color
   float ao_term = texture2D(ambientMap, texPoint.xy).r;
   finalColor = ao_term * gl_LightSource[0].ambient.xyz * texColor.xyz;//texColor.xyz * texture2D(ambientMap, texPoint.xy);//gl_LightSource[0].ambient.xyz * texColor.xyz;

   if(LDotN > 0.0 && startPoint.z / depth <= quadTreeData.x)
   {
	   if(!compute_shadow(texPoint, eyeSpaceLightDir))
	   {
 			finalColor += att * (texColor.xyz * gl_LightSource[0].diffuse.xyz * diff
 								+ gl_LightSource[0].specular.xyz * pow(spec,shininess));
	   }
   }

   projTexPoint = gl_ModelViewProjectionMatrix * vec4(texPoint.x, texPoint.y, texPoint.z, 1.0);
   projTexPoint.z /= projTexPoint.w;

   gl_FragDepth = projTexPoint.z;

   gl_FragColor = vec4(finalColor.rgb, 1.0);

   //gl_FragColor = vec4(quadTreeData.x, quadTreeData.x, quadTreeData.x,1.0);
   //gl_FragColor = vec4(gl_TexCoord[0].z / depth, gl_TexCoord[0].z / depth,gl_TexCoord[0].z / depth,1.0);
   //gl_FragColor = vec4(quadTreeData.x, quadTreeData.x, quadTreeData.x, 1.0);
   //gl_FragColor = vec4(projTexPoint.z * 0.5,projTexPoint.z * 0.5,projTexPoint.z * 0.5, 1.0);
   //gl_FragColor = vec4(eyeSpaceLightDir.xyz, 1.0);
}

float compute_exitparam(in vec3 intersectPoint, in Ray currentRay)
{
	vec3 intersectBboxMax;

	vec3 maxRayParam = currentRay.intersectBboxMax - (intersectPoint * currentRay.invRayDir);

	return min(min(maxRayParam.x, maxRayParam.y), maxRayParam.z);
}

float get_nearest_border_2D(in vec2 startPoint, in Ray currentRay, in float nodesPerRow)
{
	vec2 brickSpacePos = fract(startPoint * nodesPerRow);
	vec2 maxRayParam = currentRay.intersectBboxMax.xy - (brickSpacePos.xy * currentRay.invRayDir.xy);

	return min(maxRayParam.x, maxRayParam.y);
}

vec3 compute_normal(in vec2 texPoint)
{
	vec4 normalData = texture2D(normalMap, texPoint.xy);

	normalData.xy = normalData.xy * 2.0 - 1.0;
    normalData.z = sqrt(1.0 - dot(normalData.xy, normalData.xy));
    normalData.xyz = normalize(normalData.x * eyeSpaceTangent
							 + normalData.y * eyeSpaceBinormal
							 + normalData.z * eyeSpaceNormal);

	return normalData;
}

bool compute_shadow(in vec3 intersectPoint, in vec3 eyeSpaceLightDir)
{
	vec3 currentPos, intersectBboxMax;
	vec4 reliefData;

	// current size of search window
	float size = shadow_step_size;

	float param = 0.001;
	bool shadow = false;

	Ray lightRay;

	lightRay.rayDir = normalize(vec3(dot(eyeSpaceLightDir,  eyeSpaceTangent),
	   					   				  dot(eyeSpaceLightDir,  eyeSpaceBinormal),
   					   					  dot(-eyeSpaceLightDir, eyeSpaceNormal)));

	vec3 signRay = sign(lightRay.rayDir);
	lightRay.invRayDir  = vec3(1.0) / lightRay.rayDir;

    lightRay.intersectBboxMax.x = (signRay.x > 0.0) ? lightRay.invRayDir.x : 0.0;
    lightRay.intersectBboxMax.y = (signRay.y > 0.0) ? lightRay.invRayDir.y : 0.0;
    lightRay.intersectBboxMax.z = (signRay.z > 0.0) ? lightRay.invRayDir.z : 0.0;

	vec3 maxRayParam = lightRay.intersectBboxMax - (intersectPoint * lightRay.invRayDir);

	float endParam = min(min(maxRayParam.x, maxRayParam.y), maxRayParam.z);

	currentPos = intersectPoint + param * lightRay.rayDir;

	reliefData = texture2D(reliefMap, currentPos.xy);

	if(currentPos.z <= reliefData.x * depth)
	{
   		while(param <= endParam)
		{
			currentPos = intersectPoint + param * lightRay.rayDir;
      		reliefData = texture2D(reliefMap, currentPos.xy);

      		if (currentPos.z >= reliefData.x * depth)
      		{
         		shadow = true;
        		break;
   			}
			param += size;
		}
	}
	else
	{
		while(param <= endParam)
		{
			currentPos = intersectPoint + param * lightRay.rayDir;
      		reliefData = texture2D(reliefMap, currentPos.xy);

      		if (currentPos.z <= reliefData.x * depth)
      		{
         		shadow = true;
        		break;
   			}
			param += size;
		}
	}

	return shadow;
}

float quadtree_search(in vec3 startPoint, in Ray currentRay, in float mip_level)
{
	vec3 currentPos, exitPos;
	vec4 curQuadData, exitQuadData;

	float level = mip_level;
	float param = 0.0;
	float exitParam = 0.0;
	float maxParam = compute_exitparam(startPoint, currentRay);

	float count = 0.0;
	float maxQuadTreeSteps = 50.0;

	curQuadData = texture2DLod(quadTreeMap, startPoint.xy, level);
	// curQuadData - x = min z value,
	//				 y = max z value,
	//				 z = average


	float nodesPerRow = pow(2.0, max_mip_level - level);
///////////////////////////////////// von oben ////////////////////////////////////////
	if(currentRay.rayDir.z > 0.0)
	{
		currentPos = startPoint;
		while( level >= lowestLevel && count < maxQuadTreeSteps)
		{
			count++;

			curQuadData = texture2DLod(quadTreeMap, currentPos.xy, level);

			// wenn aktuelle position |ber hvhenfeld ist
			if(currentPos.z < curQuadData.x * depth)
			{
				// berechne schnittpunkt zum ndchstgelegenen pixel
				exitParam = get_nearest_border_2D(currentPos.xy, currentRay, nodesPerRow);

				exitPos = currentPos + exitParam * (currentRay.rayDir/vec3(nodesPerRow));

				// wenn schnittpunkt immernoch |ber dem hvhenfeld, dann betrachte ndchstes tile im gleichem miplevel
				if(exitPos.z < curQuadData.x * depth)
				{
					currentPos = exitPos + 0.0001 * currentRay.rayDir;
					param = (currentPos.x - startPoint.x) * currentRay.invRayDir.x;
				}
				// wenn schnittpunkt unter dem hvhenfeld, betrachte schnittpunkt des strahls mit dem hvhenfeld eine ebene hvher
				else
				{
					param = (curQuadData.x * depth - startPoint.z) * currentRay.invRayDir.z;
					currentPos = startPoint + param * currentRay.rayDir;
					level--;
					nodesPerRow = pow(2.0, max_mip_level - level);
				}
			 }
			 else
			 {
				 level--;
				 nodesPerRow = pow(2.0, max_mip_level - level);
			 }
		}
	}
	else
	{
		currentPos = startPoint;
		while( level >= lowestLevel && count < maxQuadTreeSteps)
		{
			count++;

			curQuadData = texture2DLod(quadTreeMap, currentPos.xy, level);

			// wenn aktuelle position unter hvhenfeld ist
			if(currentPos.z > curQuadData.a * depth)
			{
				// berechne schnittpunkt zum ndchstgelegenen pixel
				exitParam = get_nearest_border_2D(currentPos.xy, currentRay, nodesPerRow);

				exitPos = currentPos + exitParam * (currentRay.rayDir/vec3(nodesPerRow));

				// wenn schnittpunkt immernoch unter dem hvhenfeld, dann betrachte ndchstes tile im gleichem miplevel
				if(exitPos.z > curQuadData.a * depth)
				{
					currentPos = exitPos + 0.0001 * currentRay.rayDir;
					param = (currentPos.x - startPoint.x) * currentRay.invRayDir.x;
				}
				// wenn schnittpunkt |ber dem hvhenfeld, betrachte schnittpunkt des strahls mit dem hvhenfeld eine ebene hvher
				else
				{
					param = (curQuadData.a * depth - startPoint.z) * currentRay.invRayDir.z;
					currentPos = startPoint + param * currentRay.rayDir;
					level--;
					nodesPerRow = pow(2.0, max_mip_level - level);
				}
			 }
			 else
			 {
				 level--;
				 nodesPerRow = pow(2.0, max_mip_level - level);
			 }
		}
	}

	if(param > maxParam + linear_step_size) {discard;}

	return max(0.0, param - linear_step_size * 2.0);
}

float linear_search(in vec3 startPoint, in Ray currentRay, in float step_size)
{
	vec3 currentPos;
	vec4 reliefData;

	float size = step_size;

	float param = 0.0;
	float exitParam = 0.0;
    float best_param = -1.0;

	float count = 0.0;
	float maxRaySteps = 3000.0;
	float min_step = 0.001;

	reliefData = texture2D(reliefMap, startPoint.xy);
	exitParam = compute_exitparam(startPoint, currentRay) + size;

	if(startPoint.z <= reliefData.x * depth)
	{
		while(param <= exitParam)
		{
			currentPos = startPoint + param * currentRay.rayDir;
			reliefData = texture2D(reliefMap, currentPos.xy);
			count++;
			if(count > maxRaySteps)
			{
				break;
			}

			if (currentPos.z >= reliefData.x * depth)
			{
				best_param = param;   // store best param
				break;
			}

			//size = min_step + dot(compute_normal(currentPos.xy), currentRay.rayDir) * (linear_step_size - min_step);
			param += size;
		}

		if(best_param < 0.0)
		{
			currentPos = startPoint + exitParam * currentRay.rayDir;
			reliefData = texture2D(reliefMap, currentPos);

			if (currentPos.z >= reliefData.x * depth)
			{
				best_param = param;   // store best param
			}
			else
			{
				discard;
			}
		}

		// recurse around first point for closest match
		for(int i = 0; i < binary_search_steps; i++)
		{
			size *= 0.5;
			currentPos = startPoint + param * currentRay.rayDir;
			reliefData = texture2D(reliefMap ,currentPos.xy);
			if (currentPos.z >= reliefData.x * depth)
			{
				best_param = param;
				param -= 2.0 * size;
			}
			param += size;
		}
	}
	else
	{
		while(param <= exitParam)
		{
			currentPos = startPoint + param * currentRay.rayDir;
			reliefData = texture2D(reliefMap, currentPos.xy);
			count++;
			if(count > maxRaySteps)
			{
				break;
			}

			if (currentPos.z <= reliefData.x * depth)
			{
				best_param = param;   // store best param
				break;
			}

			//size = min_step + dot(compute_normal(currentPos.xy), currentRay.rayDir) * (linear_step_size - min_step);
			param += size;
		}

		if(best_param < 0.0)
		{
			currentPos = startPoint + exitParam * currentRay.rayDir;
			reliefData = texture2D(reliefMap, currentPos);

			if (currentPos.z <= reliefData.x * depth)
			{
				best_param = param;   // store best param
			}
			else
			{
				discard;
			}
		}

		// recurse around first point for closest match
		for(int i = 0; i < binary_search_steps; i++)
		{
			size *= 0.5;
			currentPos = startPoint + param * currentRay.rayDir;
			reliefData = texture2D(reliefMap ,currentPos.xy);
			if (currentPos.z <= reliefData.x * depth)
			{
				best_param = param;
				param -= 2.0 * size;
			}
			param += size;
		}
	}

	if(best_param > exitParam + linear_step_size) {discard;}

	return best_param;
}
