import qupath.ext.stardist.StarDist2D
import qupath.lib.scripting.QP

// =========================
// SETTINGS
// =========================
def atlasName = "Adult Mouse Brain - Allen Brain Atlas V3p1"
def regionNames = ["LSr", "LSc", "LSv"]

def modelPath = "I:/counting/stardist_counting_and_imagej_files/model.pb"

def crfr1Classifier = "crfr1"
def cfosClassifier  = "cfos"
def crfr2Classifier = "crfr2"

// =========================
// IMAGE DATA
// =========================
def imageData = QP.getCurrentImageData()
def rawImageName = imageData.getServer().getMetadata().getName()
def safeName = rawImageName.replaceAll("[^a-zA-Z0-9-_\\.]", "_")

// =========================
// LOAD ABBA ANNOTATIONS
// =========================
qupath.ext.biop.abba.AtlasTools.loadWarpedAtlasAnnotations(
    imageData,
    atlasName,
    "acronym",
    false,
    true
)
println("Imported ABBA annotations")

// =========================
// SELECT LS REGIONS
// =========================
def allAnnotations = QP.getAnnotationObjects()
def selectedAnnotations = allAnnotations.findAll {
    def name = it.getName()
    name != null && regionNames.contains(name)
}

if (selectedAnnotations == null || selectedAnnotations.isEmpty()) {
    println("No matching annotations (LSr, LSc, LSv) found")
    return
}

println("Found ${selectedAnnotations.size()} matching annotations: ${selectedAnnotations*.getName()}")

// =========================
// CHECK MODEL
// =========================
def modelFile = new File(modelPath)
if (!modelFile.exists()) {
    println("ERROR: Model file not found at ${modelPath}")
    return
}

// =========================
// STARDIST CELL DETECTION
// =========================
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
println("StarDist cell detection done")

// =========================
// HARALICK / INTENSITY FEATURES
// =========================
selectCells()
runPlugin(
    'qupath.lib.algorithms.IntensityFeaturesPlugin',
    '{"pixelSizeMicrons":2.0,"region":"ROI","tileSizeMicrons":25.0,"channel1":false,"channel2":true,"channel3":true,"channel4":true,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":true,"haralickMin":0.0,"haralickMax":255.0,"haralickDistance":1,"haralickBins":32}'
)
println("Haralick and intensity features calculated")

// =========================
// PREPARE CSV
// =========================
def projectDir = buildFilePath(PROJECT_BASE_DIR, "export")
mkdirs(projectDir)
def csvPath = buildFilePath(projectDir, "LS_new.csv")
def csvFile = new File(csvPath)

if (!csvFile.exists()) {
    csvFile.withWriter("UTF-8") { writer ->
        writer.writeLine(
            "IMAGENAME,ANNOTATION,DAPI,DAPI:CRFR1,DAPI:cFos,DAPI:CRFR2," +
            "DAPI:CRFR1:cFos,DAPI:CRFR2:cFos,DAPI:CRFR1:CRFR2,DAPI:CRFR1:cFos:CRFR2," +
            "PERCENT_CRFR1,PERCENT_cFos,PERCENT_CRFR2,PERCENT_CRFR1_cFos," +
            "PERCENT_CRFR2_cFos,PERCENT_CRFR1_CRFR2,PERCENT_CRFR1_cFos_CRFR2"
        )
    }
}

// =========================
// CLASSIFICATION + COUNTS
// =========================
selectedAnnotations.each { annotation ->

    println("--- Processing annotation: ${annotation.getName()} ---")

    def dapiDetections = annotation.getChildObjects()
    if (dapiDetections == null || dapiDetections.isEmpty()) {
        println("No detections found in ${annotation.getName()}")
        return
    }

    def crfr1cells = []
    def cFosCells = []
    def crfr2cells = []

    // CRFR1
    runObjectClassifier(crfr1Classifier)
    crfr1cells = dapiDetections.findAll {
        def pathClass = it.getPathClass()
        pathClass != null && pathClass.toString().toLowerCase().contains("crfr1")
    }

    // cFos
    runObjectClassifier(cfosClassifier)
    cFosCells = dapiDetections.findAll {
        def pathClass = it.getPathClass()
        pathClass != null && pathClass.toString().toLowerCase().contains("cfos")
    }

    // CRFR2
    runObjectClassifier(crfr2Classifier)
    crfr2cells = dapiDetections.findAll {
        def pathClass = it.getPathClass()
        pathClass != null && pathClass.toString().toLowerCase().contains("crfr2")
    }

    // Co-classifications
    def crfr1cFos = crfr1cells.findAll { cFosCells.contains(it) }
    def crfr2cFos = crfr2cells.findAll { cFosCells.contains(it) }
    def crfr1crfr2 = crfr1cells.findAll { crfr2cells.contains(it) }
    def crfr1cFoscrfr2 = crfr1cFos.findAll { crfr2cells.contains(it) }

    def dapiCount = dapiDetections.size().toDouble()
    def pct = { count -> dapiCount > 0 ? (count / dapiCount * 100).round(2) : 0 }

    def line = [
        safeName,
        annotation.getName(),
        dapiDetections.size(),
        crfr1cells.size(),
        cFosCells.size(),
        crfr2cells.size(),
        crfr1cFos.size(),
        crfr2cFos.size(),
        crfr1crfr2.size(),
        crfr1cFoscrfr2.size(),
        pct(crfr1cells.size()),
        pct(cFosCells.size()),
        pct(crfr2cells.size()),
        pct(crfr1cFos.size()),
        pct(crfr2cFos.size()),
        pct(crfr1crfr2.size()),
        pct(crfr1cFoscrfr2.size())
    ].join(",")

    csvFile.append(line + "\n")
    println("Saved counts and percentages for annotation: ${annotation.getName()}")
}

println("Done! Results saved in: ${csvPath}")
