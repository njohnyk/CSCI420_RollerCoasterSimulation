/*
  CSCI 420 Computer Graphics, USC
  Assignment 2: Roller Coaster
  C++ starter code

  Student username: Nikhil Johny K.
  Student ID: 2900797907
  Email: karuthed@usc.edu
*/

#include "basicPipelineProgram.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "openGLHeader.h"
#include "glutHeader.h"

#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#if defined(WIN32) || defined(_WIN32)
    #ifdef _DEBUG
        #pragma comment(lib, "glew32d.lib")
    #else
        #pragma comment(lib, "glew32.lib")
    #endif
#endif

#if defined(WIN32) || defined(_WIN32)
    char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
    char shaderBasePath[1024] = "../openGLHelper-starterCode";
#endif

using namespace std;

// Vector struct with helper methods
struct Vector {
    double x, y, z;

    Vector() { x = y = z = 0.0f; }
    
    Vector(double x, double y, double z): x(x), y(y), z(z) {}

    Vector negateVector() { 
        return scalarMultiplication(-1.0f); 
    }

    Vector addVector(Vector p) { 
        return Vector(x + p.x, y + p.y, z + p.z); 
    }

    Vector subVector(Vector p) { 
        return Vector(x - p.x, y - p.y, z - p.z); 
    }

    Vector scalarMultiplication(float scale) { 
        return Vector(scale * x, scale * y, scale * z); 
    }
    
    Vector getUnitVector() {
        double magnitude = sqrt(x * x + y * y + z * z);
        return Vector(scalarMultiplication(1.0f / magnitude));
    }

    double dotProduct(Vector p) {
        return (x * p.x + y * p.y + z * p.z);
    }

    Vector crossProduct(Vector p) {
        return Vector(y * p.z - z * p.y,
                     z * p.x - x * p.z,
                     x * p.y - y * p.x);
    }
};

// Spline struct to store spline data
struct Spline {
    int numControlPoints;
    Vector * points;
};

bool flag = false;
int counter = 0;

// Spline data
Spline * splines;
int numSplines;

// Flag to render texture shader or basic shader
bool isTextureShader;

// Mouse data
int mousePos[2]; 
int leftMouseButton = 0; 
int middleMouseButton = 0; 
int rightMouseButton = 0; 

// Transformation flags
typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;
CONTROL_STATE controlState = ROTATE;

// Initial transformations
float landRotate[3] = { 0.0f, 0.0f, 0.0f };
float landTranslate[3] = { 0.0f, 0.0f, 0.0f };
float landScale[3] = { 1.0f, 1.0f, 1.0f };

// Window data 
int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "Roller Coaster Simulation";

// Camera data
const float FOV = 45;
vector<Vector> eyePositions, fPoints, upVectors;

// Pipeline data
OpenGLMatrix matrix; 
GLuint program, textureProgram, textureHandle;
BasicPipelineProgram pipelineProgram, texPipelineProgram;
GLint modelViewMatrix, projectionMatrix, textureModelViewMatrix, textureProjectionMatrix;

// Triangle data
Vector currentTangent, currentNormal, currentB;
vector<float> normals, positions, texturePositions, textureUVs;
GLuint trackVBO, trackVAO, textureVBO, textureVAO;
int trackId = 0;

// Screenshot data
const int MAXIMUM_SCREENSHOTS = 1001;
int screenshotCounter = 0;
bool takeScreenshots = false;

// Load and store spline data from file
int loadSplines(char * argv)  {
    char * cName = (char *) malloc(128 * sizeof(char));
    FILE * fileList;
    FILE * fileSpline;
    int iType, i = 0, j, iLength;

    fileList = fopen(argv, "r");
    if (fileList == NULL) {
        printf ("can't open file\n");
        exit(1);
    }
    
    fscanf(fileList, "%d", &numSplines);

    splines = (Spline*) malloc(numSplines * sizeof(Spline));
    
    for (j = 0; j < numSplines; j++)  {
        i = 0;
        fscanf(fileList, "%s", cName);
        fileSpline = fopen(cName, "r");

        if (fileSpline == NULL) {
            printf ("can't open file\n");
            exit(1);
        }

        fscanf(fileSpline, "%d %d", &iLength, &iType);

        splines[j].points = (Vector *)malloc(iLength * sizeof(Vector));
        splines[j].numControlPoints = iLength;

        while (fscanf(fileSpline, "%lf %lf %lf",  &splines[j].points[i].x, &splines[j].points[i].y, &splines[j].points[i].z) != EOF) {
            i++;
        }
    }
    free(cName);
    return 0;
}

// Initialize texture data given image
int initTexture(const char * imageFilename, GLuint textureHandle) {
    ImageIO img;
    ImageIO::fileFormatType imgFormat;
    ImageIO::errorType err = img.load(imageFilename, &imgFormat);
    if (err != ImageIO::OK) {
        printf("Loading texture from %s failed.\n", imageFilename);
        return -1;
    }

    if (img.getWidth() * img.getBytesPerPixel() % 4) {
        printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
        return -1;
    }

    int width = img.getWidth();
    int height = img.getHeight();
    unsigned char * pixelsRGBA = new unsigned char[4 * width * height]; 

    memset(pixelsRGBA, 0, 4 * width * height); 
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            pixelsRGBA[4 * (h * width + w) + 0] = 0; 
            pixelsRGBA[4 * (h * width + w) + 1] = 0; 
            pixelsRGBA[4 * (h * width + w) + 2] = 0; 
            pixelsRGBA[4 * (h * width + w) + 3] = 255; 

            int numChannels = img.getBytesPerPixel();
            for (int c = 0; c < numChannels; c++) 
                pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
        }
    }
    
    glBindTexture(GL_TEXTURE_2D, textureHandle);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    GLfloat fLargest;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
    printf("Max available anisotropic samples: %f\n", fLargest);
    
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

    GLenum errCode = glGetError();
    if (errCode != 0) {
        printf("Texture initialization error. Error code: %d.\n", errCode);
        return -1;
    }
    
    delete [] pixelsRGBA;
    return 0;
}

// Save a screenshot
void saveScreenshot(const char * filename) {
    unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

    ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

    if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
        cout << "File " << filename << " saved successfully." << endl;
    else cout << "Failed to save file " << filename << '.' << endl;

    delete [] screenshotData;
}

// Perform Phong shading
void performPhongShading() {
    float view[16];
    matrix.SetMatrixMode(OpenGLMatrix::ModelView);
    matrix.LoadIdentity();

    // Set camera position
    matrix.LookAt(
        eyePositions.at(trackId).x,     eyePositions.at(trackId).y,     eyePositions.at(trackId).z,
        fPoints.at(trackId).x,          fPoints.at(trackId).y,          fPoints.at(trackId).z,
        upVectors.at(trackId).x,        upVectors.at(trackId).y,        upVectors.at(trackId).z);
    matrix.GetMatrix(view);
    
    // Calcuate Light direction 
    Vector lightDirection = Vector(0, 1, 0); 
    GLint loc = glGetUniformLocation(program, "viewLightDirection");
        
    Vector lD;
    lD.x = view[0] * lightDirection.x + view[4] * lightDirection.y + view[8]  * lightDirection.z;
    lD.y = view[1] * lightDirection.x + view[5] * lightDirection.y + view[9]  * lightDirection.z;
    lD.z = view[2] * lightDirection.x + view[6] * lightDirection.y + view[10] * lightDirection.z;

    float viewLightDirection[3] = { static_cast<float>(lD.x), static_cast<float>(lD.y), static_cast<float>(lD.z) };
    glUniform3fv(loc, 1, viewLightDirection);

    // Set up Phong shading coefficients and values
    float lightIntensity = 0.7f;
    float materialCoefficient = 0.9f;

    loc = glGetUniformLocation(program, "La");
    float La[4] = { lightIntensity, lightIntensity, lightIntensity, 1.0f };
    glUniform4fv(loc, 1, La);

    loc = glGetUniformLocation(program, "ka");
    float ka[4] = { materialCoefficient, 0, 0, 0.2 };
    glUniform4fv(loc, 1, ka);

    loc = glGetUniformLocation(program, "Ld");
    float Ld[4] = { lightIntensity, lightIntensity, lightIntensity, 1.0f };
    glUniform4fv(loc, 1, Ld);

    loc = glGetUniformLocation(program, "kd");
    float kd[4] = { materialCoefficient, 0, 0, 0.5 };
    glUniform4fv(loc, 1, kd);

    loc = glGetUniformLocation(program, "Ls");
    float Ls[4] = { lightIntensity, lightIntensity, lightIntensity, lightIntensity };
    glUniform4fv(loc, 1, Ls);

    loc = glGetUniformLocation(program, "ks");
    float ks[4] = { materialCoefficient, 0, 0, materialCoefficient };
    glUniform4fv(loc, 1, ks);

    loc = glGetUniformLocation(program, "alpha");
    glUniform1f(loc, 1.0f);

    // Set up model, perspective and normal matrix
    float m[16]; 
    matrix.GetMatrix(m);
    glUniformMatrix4fv(modelViewMatrix, 1, GL_FALSE, m);
        
    matrix.SetMatrixMode(OpenGLMatrix::Projection);
    float p[16]; 
    matrix.GetMatrix(p);
    glUniformMatrix4fv(projectionMatrix, 1, GL_FALSE, p);

    GLint normalMatrix = glGetUniformLocation(program, "normalMatrix");
    float n[16];
    matrix.SetMatrixMode(OpenGLMatrix::ModelView);
    matrix.GetNormalMatrix(n);
    glUniformMatrix4fv(normalMatrix, 1, GL_FALSE, n);
}

// Calculate for mode 1: p(u) = [u^3 u^2 u 1] * M * C
// Calculate for mode 2: t(u) = [3u^2 2u 1 0] * M * C
Vector matrixSplineMultiplication(int modeType, float u, float s, Vector P1, Vector P2, Vector P3, Vector P4) {
    Vector resultVector;
    float MC[12];
    float M[16] = { 
        -s,         2.0f - s,   s - 2.0f,           s,
        2.0f * s,   s - 3.0f,   3.0f - 2.0f * s,    -s,
        -s,         0.0f,       s,                  0.0f,
         0.0f,      1.0f,       0.0f,               0.0f 
     };

    float uMatrix[4];
    if (modeType == 1) {
        uMatrix[0] = u * u * u; uMatrix[1] = u * u; uMatrix[2] = u; uMatrix[3] = 1.0f;
    }
    else if (modeType == 2) {
        uMatrix[0] = 3 * u * u; uMatrix[1] = 2 * u; uMatrix[2] = 1.0f; uMatrix[3] = 0.0f;
    }

    MC[0] = M[0] * P1.x + M[1] * P2.x + M[2] * P3.x + M[3] * P4.x;
    MC[1] = M[0] * P1.y + M[1] * P2.y + M[2] * P3.y + M[3] * P4.y;
    MC[2] = M[0] * P1.z + M[1] * P2.z + M[2] * P3.z + M[3] * P4.z;

    MC[3] = M[4] * P1.x + M[5] * P2.x + M[6] * P3.x + M[7] * P4.x;
    MC[4] = M[4] * P1.y + M[5] * P2.y + M[6] * P3.y + M[7] * P4.y;
    MC[5] = M[4] * P1.z + M[5] * P2.z + M[6] * P3.z + M[7] * P4.z;

    MC[6] = M[8] * P1.x + M[9] * P2.x + M[10] * P3.x + M[11] * P4.x;
    MC[7] = M[8] * P1.y + M[9] * P2.y + M[10] * P3.y + M[11] * P4.y;
    MC[8] = M[8] * P1.z + M[9] * P2.z + M[10] * P3.z + M[11] * P4.z;

    MC[9]  = M[12] * P1.x + M[13] * P2.x + M[14] * P3.x + M[15] * P4.x;
    MC[10] = M[12] * P1.y + M[13] * P2.y + M[14] * P3.y + M[15] * P4.y;
    MC[11] = M[12] * P1.z + M[13] * P2.z + M[14] * P3.z + M[15] * P4.z;

    resultVector.x = uMatrix[0] * MC[0] + uMatrix[1] * MC[3] + uMatrix[2] * MC[6] + uMatrix[3] * MC[9];
    resultVector.y = uMatrix[0] * MC[1] + uMatrix[1] * MC[4] + uMatrix[2] * MC[7] + uMatrix[3] * MC[10];
    resultVector.z = uMatrix[0] * MC[2] + uMatrix[1] * MC[5] + uMatrix[2] * MC[8] + uMatrix[3] * MC[11];

    return resultVector;
}

// Create a triangle
void createTriangle(Vector v1, Vector v2, Vector v3, Vector normal){
    // Vertex 1 and Normals
    positions.push_back(v1.x);
    positions.push_back(v1.y);
    positions.push_back(v1.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);

    // Vertex 2 and Normals
    positions.push_back(v2.x);
    positions.push_back(v2.y);
    positions.push_back(v2.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);

    // Vertex 3 and Normals
    positions.push_back(v3.x);
    positions.push_back(v3.y);
    positions.push_back(v3.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);
}

// Get vertex position
Vector getVertexPosition(Vector p, float scale, Vector N, Vector B) {
    return p.addVector(N.addVector(B).scalarMultiplication(scale));
}

// Create crossBars
void createCrossBar(Vector v1, Vector v2, Vector v3, Vector v4, Vector v5, Vector v6, Vector normal){
    // Vertex 1 and Normals
    positions.push_back(v1.x);
    positions.push_back(v1.y);
    positions.push_back(v1.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);

    // Vertex 2 and Normals
    positions.push_back(v2.x);
    positions.push_back(v2.y);
    positions.push_back(v2.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);

    // Vertex 3 and Normals
    positions.push_back(v3.x);
    positions.push_back(v3.y);
    positions.push_back(v3.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);

    // Vertex 4 and Normals
    positions.push_back(v4.x);
    positions.push_back(v4.y);
    positions.push_back(v4.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);

    // Vertex 5 and Normals
    positions.push_back(v5.x);
    positions.push_back(v5.y);
    positions.push_back(v5.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);

    // Vertex 6 and Normals
    positions.push_back(v6.x);
    positions.push_back(v6.y);
    positions.push_back(v6.z);
    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);
}

// Create left and right track for the roller coaster
void createRollerCoasterTracks(float u, int controlPoint) {
    Vector p0, p1;
    Vector lt0, lt1, lt2, lt3, lt4, lt5, lt6, lt7;
    Vector rt0, rt1, rt2, rt3, rt4, rt5, rt6, rt7;
    float trackWidth = 0.05f;
    float trackInterval = 1.0f;

    p0 = matrixSplineMultiplication(1, u, 0.5f, splines[0].points[controlPoint], splines[0].points[controlPoint+1], splines[0].points[controlPoint+2], splines[0].points[controlPoint+3]);

    // Get next 4 control points
    Vector P1, P2, P3, P4;
    P1 = splines[0].points[controlPoint];
    P2 = splines[0].points[controlPoint+1];
    P3 = splines[0].points[controlPoint+2];
    P4 = splines[0].points[controlPoint+3];

    // Calculate tangent, normal and B
    currentTangent = matrixSplineMultiplication(2, u, 0.5f, P1, P2, P3, P4).getUnitVector();
    if (u == 0.00f && controlPoint == 0) {
        currentNormal = currentTangent.crossProduct(Vector(0.0f, -1.0f, 0.0f)).getUnitVector();
    }
    else {
        currentNormal = currentB.crossProduct(currentTangent).getUnitVector(); 
    }
    currentB = currentTangent.crossProduct(currentNormal).getUnitVector();

    // Set camera position
    float viewFactor = 0.5f;
    Vector p = p0.addVector(currentB.scalarMultiplication(viewFactor * trackInterval));
    eyePositions.push_back(p.addVector(currentNormal));
    fPoints.push_back(p.addVector(currentNormal).addVector(currentTangent));
    upVectors.push_back(currentNormal);

    // Calculate left and right track position
    lt0 = getVertexPosition(p0, trackWidth, Vector(), currentB);
    lt1 = getVertexPosition(p0, trackWidth, currentNormal, currentB);
    lt2 = getVertexPosition(p0, trackWidth, currentNormal, currentB.negateVector());
    lt3 = getVertexPosition(p0, trackWidth, Vector(), currentB.negateVector());

    rt0 = getVertexPosition(p0.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, Vector(), currentB);
    rt1 = getVertexPosition(p0.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, currentNormal, currentB);
    rt2 = getVertexPosition(p0.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, currentNormal, currentB.negateVector());
    rt3 = getVertexPosition(p0.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, Vector(), currentB.negateVector());

    p1 = matrixSplineMultiplication(1, u+0.01f, 0.5f, splines[0].points[controlPoint], splines[0].points[controlPoint+1], splines[0].points[controlPoint+2], splines[0].points[controlPoint+3]);

    // Get next 4 control points
    float newU = u + 0.01f;
    P1 = splines[0].points[controlPoint];
    P2 = splines[0].points[controlPoint + 1];
    P3 = splines[0].points[controlPoint + 2];
    P4 = splines[0].points[controlPoint + 3];

    // Calculate tangent, normal and B
    currentTangent = matrixSplineMultiplication(2, newU, 0.5f, P1, P2, P3, P4).getUnitVector();
    if (newU == 0.00f && controlPoint == 0) {
        currentNormal = currentTangent.crossProduct(Vector(0.0f, -1.0f, 0.0f)).getUnitVector();
    }
    else {
        currentNormal = currentB.crossProduct(currentTangent).getUnitVector(); 
    }
    currentB = currentTangent.crossProduct(currentNormal).getUnitVector();

    // Calculate left and right track position
    lt4 = getVertexPosition(p1, trackWidth, Vector(), currentB);
    lt5 = getVertexPosition(p1, trackWidth, currentNormal, currentB);
    lt6 = getVertexPosition(p1, trackWidth, currentNormal, currentB.negateVector());
    lt7 = getVertexPosition(p1, trackWidth, Vector(), currentB.negateVector());

    rt4 = getVertexPosition(p1.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, Vector(), currentB);
    rt5 = getVertexPosition(p1.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, currentNormal, currentB);
    rt6 = getVertexPosition(p1.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, currentNormal, currentB.negateVector());
    rt7 = getVertexPosition(p1.addVector(currentB.scalarMultiplication(trackInterval)), trackWidth, Vector(), currentB.negateVector());

    // Create triangles for left track
    createTriangle(lt0, lt1, lt3, currentTangent.negateVector()); 
    createTriangle(lt2, lt1, lt3, currentTangent.negateVector());
    createTriangle(lt1, lt0, lt5, currentB);       
    createTriangle(lt4, lt0, lt5, currentB);
    createTriangle(lt2, lt1, lt6, currentNormal);       
    createTriangle(lt5, lt1, lt6, currentNormal);
    createTriangle(lt3, lt2, lt7, currentB.negateVector()); 
    createTriangle(lt6, lt2, lt7, currentB.negateVector());
    createTriangle(lt3, lt0, lt7, currentNormal.negateVector()); 
    createTriangle(lt4, lt0, lt7, currentNormal.negateVector());
    createTriangle(lt5, lt4, lt6, currentTangent);       
    createTriangle(lt7, lt4, lt6, currentTangent);

    // Create triangles for right track
    createTriangle(rt0, rt1, rt3, currentTangent.negateVector());  
    createTriangle(rt2, rt1, rt3, currentTangent.negateVector());
    createTriangle(rt1, rt0, rt5, currentB);        
    createTriangle(rt4, rt0, rt5, currentB);
    createTriangle(rt2, rt1, rt6, currentNormal);        
    createTriangle(rt5, rt1, rt6, currentNormal);
    createTriangle(rt3, rt2, rt7, currentB.negateVector());  
    createTriangle(rt6, rt2, rt7, currentB.negateVector());
    createTriangle(rt3, rt0, rt7, currentNormal.negateVector());  
    createTriangle(rt4, rt0, rt7, currentNormal.negateVector());
    createTriangle(rt5, rt4, rt6, currentTangent);        
    createTriangle(rt7, rt4, rt6, currentTangent);

    // Create cross bars
    if(flag) {
        createCrossBar(lt4, rt4, lt0, rt0, lt0, rt4, currentB);
    }   
}

// Render Roller Coaster Track
void renderRollerCoasterTrack(char *argv[]) {
    // Load spline data
    loadSplines(argv[1]);
    printf("Loaded %d spline(s).\n", numSplines);
    for(int i = 0; i < numSplines; i++) {
        printf("Num control points in spline %d: %d.\n", i, splines[i].numControlPoints);
    }

    // Add triangles to the splines to create track mesh
    for (int controlPoint = 0; controlPoint < splines[0].numControlPoints - 3; controlPoint++) {
        for (float u = 0.0f; u <= 1.0f; u += 0.01f) {
            createRollerCoasterTracks(u, controlPoint);
            counter++;
            if(counter % 4 == 0) {
                flag = true;
            }
            else {
                flag = false;
            }
        }
    }

    // Set up pipeline
    isTextureShader = true;
    pipelineProgram.Init(shaderBasePath, isTextureShader);
    pipelineProgram.Bind(); 
    program = pipelineProgram.GetProgramHandle();
    modelViewMatrix = glGetUniformLocation(program, "modelViewMatrix");
    projectionMatrix = glGetUniformLocation(program, "projectionMatrix");

    // Set up VBOs
    glGenBuffers(1, &trackVBO);
    glBindBuffer(GL_ARRAY_BUFFER, trackVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (positions.size() + normals.size()), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * positions.size(), static_cast<void*>(positions.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * positions.size(), sizeof(float) * normals.size(), static_cast<void*>(normals.data()));

    // Set up VAOs
    glGenVertexArrays(1, &trackVAO);
    glBindVertexArray(trackVAO); 

    GLuint loc = glGetAttribLocation(program, "position"); 
    glEnableVertexAttribArray(loc); 
    const void * offset = (const void*) 0;
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, offset);

    loc = glGetAttribLocation(program, "normal"); 
    glEnableVertexAttribArray(loc); 
    offset = (const void*) (sizeof(float) * positions.size());
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, offset);
}

// Render Ground Texture
void renderGroundTexture() {
    float size = 1000.0f;
    float height = 20.0f;

    texturePositions.push_back(size);  texturePositions.push_back(-size); texturePositions.push_back(height);
    texturePositions.push_back(size);  texturePositions.push_back(size);  texturePositions.push_back(height);
    texturePositions.push_back(-size); texturePositions.push_back(size);  texturePositions.push_back(height);

    texturePositions.push_back(-size); texturePositions.push_back(size);  texturePositions.push_back(height);
    texturePositions.push_back(-size); texturePositions.push_back(-size); texturePositions.push_back(height);
    texturePositions.push_back(size);  texturePositions.push_back(-size); texturePositions.push_back(height);

    float scale = 200.0f;
    textureUVs.push_back(scale); textureUVs.push_back(scale);
    textureUVs.push_back(scale); textureUVs.push_back(0.0f);
    textureUVs.push_back(0.0f);  textureUVs.push_back(0.0f);

    textureUVs.push_back(0.0f);  textureUVs.push_back(0.0f);
    textureUVs.push_back(0.0f);  textureUVs.push_back(scale);
    textureUVs.push_back(scale); textureUVs.push_back(scale);

    isTextureShader = false;
    texPipelineProgram.Init(shaderBasePath, isTextureShader);
    texPipelineProgram.Bind(); 
    
    textureProgram = texPipelineProgram.GetProgramHandle();
    textureModelViewMatrix = glGetUniformLocation(textureProgram, "modelViewMatrix");
    textureProjectionMatrix = glGetUniformLocation(textureProgram, "projectionMatrix");

    // Set up ground texture
    glGenTextures(1, &textureHandle);
    initTexture("./textures/groundTexture.jpg", textureHandle);

    // Set up texture trackVBO
    glGenBuffers(1, &textureVBO);
    glBindBuffer(GL_ARRAY_BUFFER, textureVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (texturePositions.size() + textureUVs.size()), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * texturePositions.size(), static_cast<void*>(texturePositions.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * texturePositions.size(), sizeof(float) * textureUVs.size(), static_cast<void*>(textureUVs.data()));

    // Set up texture trackVAO
    glGenVertexArrays(1, &textureVAO);
    glBindVertexArray(textureVAO); 

    GLuint loc = glGetAttribLocation(textureProgram, "position"); 
    glEnableVertexAttribArray(loc); 
    const void * offset = (const void*) 0;
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, offset);

    loc = glGetAttribLocation(textureProgram, "textureCoordinates"); 
    glEnableVertexAttribArray(loc); 
    offset = (const void*) (sizeof(float) * texturePositions.size());
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, offset);
}

//  Render tracks, texture and perform Phong shading
void displayFunc() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pipelineProgram.Bind();

    // perform Phong shading
    performPhongShading(); 

    // Render track
    glBindVertexArray(trackVAO); 
    glDrawArrays(GL_TRIANGLES, 0, positions.size());

    // Render texture
    glActiveTexture(GL_TEXTURE0);
    GLint textureImage = glGetUniformLocation(textureProgram, "textureImage");
    glUniform1i(textureImage, GL_TEXTURE0 - GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureHandle);
    texPipelineProgram.Bind();

    // Set up model view matrix
    matrix.SetMatrixMode(OpenGLMatrix::ModelView);
    matrix.LoadIdentity();
    matrix.LookAt(
        eyePositions.at(trackId).x,     eyePositions.at(trackId).y,     eyePositions.at(trackId).z,
        fPoints.at(trackId).x,          fPoints.at(trackId).y,          fPoints.at(trackId).z,
        upVectors.at(trackId).x,        upVectors.at(trackId).y,        upVectors.at(trackId).z
    );

    // Set up textures matrices
    float m[16]; 
    matrix.GetMatrix(m);
    glUniformMatrix4fv(textureModelViewMatrix, 1, GL_FALSE, m);
    matrix.SetMatrixMode(OpenGLMatrix::Projection);

    float p[16]; 
    matrix.GetMatrix(p);
    glUniformMatrix4fv(textureProjectionMatrix, 1, GL_FALSE, p);
    
    // Render texture
    glBindVertexArray(textureVAO); 
    glDrawArrays(GL_TRIANGLES, 0, texturePositions.size());
    glBindVertexArray(0); 
    glutSwapBuffers();    
}

// Save animation if flag is true
// Increment camera position
void idleFunc() {
    // Save animation
    if (screenshotCounter < MAXIMUM_SCREENSHOTS && takeScreenshots) {
        stringstream ss;
        ss << "anim/" << screenshotCounter << ".jpg";
        string name = ss.str();
        char const* pchar = name.c_str();
        saveScreenshot(pchar);
        screenshotCounter++;
    }
    trackId += 2;
    if (trackId == eyePositions.size()) {
        trackId = 0;
    }
    glutPostRedisplay();
}

// Reset viewport and set perspective matrix
void reshapeFunc(int w, int h) {
    glViewport(0, 0, w, h);
    matrix.SetMatrixMode(OpenGLMatrix::Projection);
    matrix.LoadIdentity();
    matrix.Perspective(FOV, (float)windowWidth / (float)windowHeight, 0.01, 1000.0);
}

// Detect mouse for translation, rotation and scaling
void mouseMotionDragFunc(int x, int y) {
    int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };
    switch (controlState) {
        case TRANSLATE:
            if (leftMouseButton) {
                landTranslate[0] += mousePosDelta[0] * 0.01f;
                landTranslate[1] -= mousePosDelta[1] * 0.01f;
            }
            if (middleMouseButton) {
                landTranslate[2] += mousePosDelta[1] * 0.01f;
            }
            break;
        case ROTATE:
            if (leftMouseButton) {
                landRotate[0] += mousePosDelta[1];
                landRotate[1] += mousePosDelta[0];
            }
            if (middleMouseButton){
                landRotate[2] += mousePosDelta[1];
            }
            break;
        case SCALE:
            if (leftMouseButton) {
                landScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
                landScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
            }
            if (middleMouseButton) {
                landScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
            }
            break;
    }
    mousePos[0] = x;
    mousePos[1] = y;
}

// Store x and y position of mouse
void mouseMotionFunc(int x, int y) {
    mousePos[0] = x;
    mousePos[1] = y;
}

// Set flag depending on mouse button press
void mouseButtonFunc(int button, int state, int x, int y) {
    switch (button) {
    case GLUT_LEFT_BUTTON:
        leftMouseButton = (state == GLUT_DOWN);
        break;

    case GLUT_MIDDLE_BUTTON:
        middleMouseButton = (state == GLUT_DOWN);
        break;

    case GLUT_RIGHT_BUTTON:
        rightMouseButton = (state == GLUT_DOWN);
        break;
    }

    mousePos[0] = x;
    mousePos[1] = y;
}

// Assign keyboard keys to modes and transformations
void keyboardFunc(unsigned char key, int x, int y) {
    int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };
    switch (key) {
        case 27: 
            exit(0); 
            break;
        case 'x':
            // Take a single screenshot
            saveScreenshot("screenshot.jpg");
            break;
        case 't':
            controlState = TRANSLATE;
            break;
        case 'a':
            // Start animation
            takeScreenshots = true;
            cout << "Starting animation!." << endl;
            break;
    }
    mousePos[0] = x;
    mousePos[1] = y;
}

// Initialize scene
void initScene(int argc, char *argv[]) {
    // Background color
    glClearColor(1.0f, 1.0f, 0.8f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // Render roller coaster and ground texture
    renderRollerCoasterTrack(argv);    
    renderGroundTexture();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf ("usage: %s <trackfile>\n", argv[0]);
        exit(0);
    }

    cout << "Initializing GLUT..." << endl;
    glutInit(&argc,argv);
    cout << "Initializing OpenGL..." << endl;

    #ifdef __APPLE__
        glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    #else
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    #endif

    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(0, 0);    
    glutCreateWindow(windowTitle);

    cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
    cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
    cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

    // tells glut to use a particular display function to redraw 
    glutDisplayFunc(displayFunc);
    // perform animation inside idleFunc
    glutIdleFunc(idleFunc);
    // callback for mouse drags
    glutMotionFunc(mouseMotionDragFunc);
    // callback for idle mouse movement
    glutPassiveMotionFunc(mouseMotionFunc);
    // callback for mouse button changes
    glutMouseFunc(mouseButtonFunc);
    // callback for resizing the window
    glutReshapeFunc(reshapeFunc);
    // callback for pressing the keys on the keyboard
    glutKeyboardFunc(keyboardFunc);

    // init glew
    #ifdef __APPLE__
    // nothing is needed on Apple
    #else
        // Windows, Linux
        GLint result = glewInit();
        if (result != GLEW_OK) {
            cout << "error: " << glewGetErrorString(result) << endl;
            exit(EXIT_FAILURE);
        }
    #endif

    // do initialization
    initScene(argc, argv);

    // sink forever into the glut loop
    glutMainLoop();
}