//Import List***
import qupath.lib.measurements.MeasurementList
import qupath.lib.objects.PathAnnotationObject
import qupath.ext.biop.cellpose.Cellpose2D

//ABBA Annotations***
qupath.ext.biop.abba.AtlasTools.loadWarpedAtlasAnnotations(
    getCurrentImageData(), 
    "Adult Mouse Brain - Allen Brain Atlas V3p1", 
    "acronym", 
    false, 
    true
)

println("Imported ABBA Annotations")

def rootAnnotation = getAnnotationObjects().find { it.getName()?.toLowerCase() == "root" }

if (rootAnnotation == null) {
    println("No root found")
    return
}

println("Defined Root")

//Definitions***
def imageData = getCurrentImageData()
def annotations = getAnnotationObjects()
def pathObjects = getAnnotationObjects()

if (pathObjects.isEmpty()) {
    println("No Annotations found")
    return
}

println("Definitions done")

//Cellpose DAPI***
def pathModeldapi = "G:/counting/cfos_whole_brain/cellpose_model_dapi_1024px"

def cellposeDAPI = Cellpose2D.builder(pathModeldapi)
          .pixelSize(0.5)
          .channels(0)
          .tileSize(1024)
          .setOverlap(128)
          .cellprobThreshold(0.5)
          .flowThreshold(0.0)
          .measureShape()
          .measureIntensity()
          .build()

import qupath.lib.roi.RoiTools

def server = imageData.getServer()
def imageWidth = server.getWidth()
def imageHeight = server.getHeight()
def imageROI = qupath.lib.roi.ROIs.createRectangleROI(0, 0, imageWidth, imageHeight)
def croppedROI = qupath.lib.roi.RoiTools.combineROIs(rootAnnotation.getROI(), imageROI, RoiTools.CombineOp.INTERSECT)

if (croppedROI == null) {
    println("The root annotation is completely outside the image and will be removed")
    removeObject(rootAnnotation)
    return
} else {
    rootAnnotation.setROI(croppedROI)
    println("Root annotation cropped to image area")
}

def fullImageROI = qupath.lib.roi.ROIs.createRectangleROI(0, 0, imageWidth, imageHeight)
def fullImageAnnotation = new PathAnnotationObject(fullImageROI)
cellposeDAPI.detectObjects(imageData, [fullImageAnnotation])
def dapiDetections = fullImageAnnotation.getChildObjects().collect()

println("DAPI detected over full image")

addObjects(dapiDetections)
fireHierarchyUpdate()
selectObjects(dapiDetections)
println("Cells stay visible in GUI")

//Haralick Calculations***
selectDetections()
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons":1.0,"region":"ROI","tileSizeMicrons":25.0,"channel1":false,"channel2":true,"channel3":true,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":true,"haralickMin":0.0,"haralickMax":255.0,"haralickDistance":1,"haralickBins":32}')
selectDetections()
runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin', '{"fwhmMicrons":20.0,"smoothWithinClasses":false}')

//Classifier cFos***
println("Run cFosClassifier")
runObjectClassifier("cfos")
println("cFosClassifier done")

//Count***
def dapiTotal = dapiDetections.size()
println("Set DAPI total")

def dapiCfosPositive = dapiDetections.findAll {
    def pathClass = it.getPathClass()
    return pathClass != null && pathClass.toString().toLowerCase().contains("cfos")
}

def dapiCfosCount = dapiCfosPositive.size()
println("Set cFos")

def cfosFraction = dapiTotal > 0 ? (dapiCfosCount / (double)dapiTotal) * 100 : 0.0
println("Calculating percentage...")

println "DAPI: ${dapiTotal}"
println "DAPI:cFos: ${dapiCfosCount}"
println "Percentage: ${cfosFraction.round(2)}%"

//Assign to Annotations + save as .csv***
qupath.ext.biop.abba.AtlasTools.loadWarpedAtlasAnnotations(
    getCurrentImageData(), 
    "Adult Mouse Brain - Allen Brain Atlas V3p1", 
    "acronym", 
    false, 
    true
)

println("Re-Imported ABBA Annotations")

def rawName = QP.getCurrentImageData().getServer().getMetadata().getName()
def safeName = rawName.replaceAll("[^a-zA-Z0-9-_\\.]", "_")

def csvPath = "G:/results_M?.csv"
def file = new File(csvPath)

def fileExists = file.exists()

file.withWriterAppend { writer ->

    if (!fileExists) {
        writer.writeLine("Image,ROI,DAPI,DAPI:cFos")
    }

    def abbaAnnotations = getAnnotationObjects().findAll { it != rootAnnotation }

    for (def annotation : abbaAnnotations) {
        def name = annotation.getName() ?: "Annotation_" + abbaAnnotations.indexOf(annotation)

        def containedDAPI = dapiDetections.findAll {
            annotation.getROI().contains(it.getROI().getCentroidX(), it.getROI().getCentroidY())
        }

        def colocalizedCount = containedDAPI.count {
            def pathClass = it.getPathClass()
            pathClass != null && pathClass.toString().toLowerCase().contains("cfos")
        }

        writer.writeLine("$safeName,$name,${containedDAPI.size()},$colocalizedCount")
        println("Saved: " + name)
    }
}

println("Batch done.")
