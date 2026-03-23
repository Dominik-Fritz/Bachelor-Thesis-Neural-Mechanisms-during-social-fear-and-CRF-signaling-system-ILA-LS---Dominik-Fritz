//Import***

import qupath.ext.stardist.StarDist2D
import qupath.lib.scripting.QP



//Put ABBA Annotations on the images in project***


qupath.ext.biop.abba.AtlasTools.loadWarpedAtlasAnnotations(
    getCurrentImageData(), 
    "Adult Mouse Brain - Allen Brain Atlas V3p1", 
    "acronym", 
    false, 
    true);
    
println( "Imported ABBA Annotations" )


//Definitions for annotations, model path and image data***

def allAnnotations = QP.getAnnotationObjects()
def selectedAnnotations = allAnnotations.findAll {
    def name = it.getName()
    return name == "ILA"
}

if (selectedAnnotations == null || selectedAnnotations.isEmpty()) {
    println("No matching Annotation found")
    return
}

println("Found ${selectedAnnotations.size()} matching Annotations")

def modelPath = "G:/counting/stardist_counting_and_imagej_files/model.pb"
def modelFile = new File(modelPath)
if (!modelFile.exists()) {
    print("ERROR: Model file not found at ${modelPath}")
    return
}

def imageData = QP.getCurrentImageData()



//Stardist Cell Detection - DAPI channel 0***

def stardist = StarDist2D
    .builder(modelPath)
    .channels(0)
    .normalizePercentiles(1, 99)
    .threshold(0.5)
    .pixelSize(0.9)
    .cellExpansion(5)
    .measureShape()
    .measureIntensity()
    .build()

stardist.detectObjects(imageData, selectedAnnotations)
stardist.close()
println("Stardist Cell Detection done")



//Haralick Calculation***

selectDetections();
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons":2.0,"region":"ROI","tileSizeMicrons":25.0,"channel1":false,"channel2":true,"channel3":true,"channel4":true,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":true,"haralickMin":0.0,"haralickMax":255.0,"haralickDistance":1,"haralickBins":32}')



//Classify and Save***

def rawImageName = imageData.getServer().getMetadata().getName()
def safeName = rawImageName.replaceAll("[^a-zA-Z0-9-_\\.]", "_")

def crfClassifier = "CRF" 
def cfosClassifier  = "cfos" 

def projectDir = buildFilePath(PROJECT_BASE_DIR, "export")
mkdirs(projectDir)
def csvPath = buildFilePath(projectDir, "ILA.csv")
def csvFile = new File(csvPath)

if (!csvFile.exists()) {
    csvFile.withWriter('UTF-8') { writer ->
        writer.writeLine("IMAGENAME,ANNOTATION,DAPI,DAPI:CRF,DAPI:cFos,DAPI:CRF:cFos," +
                         "PERCENT_CRF,PERCENT_cFos,PERCENT_CRF_cFos")
    }
}

selectedAnnotations.each { annotation ->

    println("--- Processing annotation: ${annotation.getName()} ---")

    def dapiDetections = annotation.getChildObjects()

    def CRFcells = []
    def cFosCells = []

    runObjectClassifier(crfClassifier)
    CRFcells = dapiDetections.findAll {
        def pathClass = it.getPathClass()
        pathClass != null && pathClass.toString().toLowerCase().contains("crf")
    }

    runObjectClassifier(cfosClassifier)
    cFosCells = dapiDetections.findAll {
        def pathClass = it.getPathClass()
        pathClass != null && pathClass.toString().toLowerCase().contains("cfos")
    }


    def CRFcFos = CRFcells.findAll { cFosCells.contains(it) }

    def dapiCount = dapiDetections.size().toDouble()
    def pct = { count -> dapiCount > 0 ? (count / dapiCount * 100).round(2) : 0 }

    def line = [
        safeName,
        annotation.getName(),
        dapiDetections.size(),
        CRFcells.size(),
        cFosCells.size(),
        CRFcFos.size(),
        pct(CRFcells.size()),
        pct(cFosCells.size()),
        pct(CRFcFos.size()),
    ].join(",")

    csvFile.append(line + "\n")
    println("Saved counts and percentages for annotation: ${annotation.getName()}")
}

println("Done! Results saved in: ${csvPath}")



