package catbreed.identification

import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.support.v7.app.AppCompatActivity
import android.util.TypedValue
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import android.util.Log
import com.opencsv.CSVReader;
import java.io.IOException;
import java.io.FileReader;

import catbreed.identification.R
import java.io.BufferedReader
import java.io.FileInputStream

/**
 * The main and only activity of the food identifier app.
 */
class MainActivity : AppCompatActivity() {

    // This is needed to request camera permissions, but it can be any value.
    private val CAMERA_REQUEST_CODE = 12345

    private var cameraPreview: TextureView? = null

    // These are all needed to use the camera. The CameraManager is needed to
    // access the cameras of the device. The cameraId stores the identifier of
    // the backward-facing camera. The CameraDevice is the actual camera that
    // can be used to take pictures. And the CameraCaptureSession is a session
    // with the camera that needs to be started before image capture is possible.
    private var cameraManager: CameraManager? = null
    private var cameraId: String? = null
    private var camera: CameraDevice? = null
    private var cameraCaptureSession: CameraCaptureSession? = null

    private var detector: CatBreedIdentifier? = null

    // The camera actions don't need to be performed in the UI thread, so there
    // will be a background thread for camera access.
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private val label_map = HashMap<Int, String>() //using a sparse array is probably the better option

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // This needs to be in onCreate, since it's not possible to access system
        // service before onCreate is called.
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        detector = CatBreedIdentifier(this)
        setContentView(R.layout.activity_main)

        // The camera permission is listed in the manifest, but the app still needs
        // to get the user's explicit permission. If it's not yet present, ask for it.
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)
        }

        // Set the field for the preview area of the camera.
        cameraPreview = findViewById(R.id.camera_preview)

        // Set label map
        setLabels(this) //is "this" a context?

        // Set up click handling for the two buttons of the interface
        val evaluateButton = setupEvaluateButton()
        setupResetButton(evaluateButton)
    }

    private fun setupEvaluateButton(): Button {
        val evaluateButton = findViewById<Button>(R.id.evaluate_button)
        evaluateButton.setOnClickListener {
            // Clicking the Evaluate button triggers cuteness detection. First, extract
            // the complete camera preview area as a bitmap
            val fullBitmap = cameraPreview!!.bitmap

            // The rectangle that shows the detection area is 299 dp in size. To get this
            // area from the bitmap, the dp value first needs to be converted to pixels,
            // and then that amount of pixels needs to be extracted from the center of the
            // full preview bitmap.
            val px = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 299f, resources.displayMetrics).toInt()
            val left_x = (fullBitmap.width - px) / 2
            val top_y = (fullBitmap.height - px) / 2
            val croppedBitmap = Bitmap.createBitmap(fullBitmap, left_x, top_y, px, px)

            // Using the captured_image ImageView the app will freeze the captured image
            // on screen so the user can move the phone and keep seeing what exact image
            // is being evaluated.
            val capturedImage = findViewById<ImageView>(R.id.captured_image)
            capturedImage.visibility = View.VISIBLE
            capturedImage.setImageBitmap(croppedBitmap)

            // Hide the evaluate button while the evaluation is running. During the
            // evaluation there is no possibility for the user to interact with the app.
            val evaluateButton = findViewById<Button>(R.id.evaluate_button)
            evaluateButton.visibility = View.GONE

            // Call the detector with the captured image and process the result in
            // the UI thread.
            detector!!.catBreedPercentage(croppedBitmap) { catid ->
                runOnUiThread {
                    setCatIdentity(catid)
                }
            }
        }
        return evaluateButton
    }

    private fun setupResetButton(evaluateButton: Button) {
        val resetButton = findViewById<Button>(R.id.reset_button)
        resetButton.setOnClickListener {
            // Clicking the reset button will return things back to the initial
            // state: Only the Evaluate button is visible, with the camera preview
            // still running, but all other UI elements are gone.
            evaluateButton.visibility = View.VISIBLE
            resetButton.visibility = View.GONE
            val capturedImage = findViewById<ImageView>(R.id.captured_image)
            val catBreedPercentageText = findViewById<TextView>(R.id.catbreed_percentage)
            val catBreedIdentity = findViewById<TextView>(R.id.catbreed_label)

            capturedImage.visibility = View.GONE
            catBreedPercentageText.visibility = View.GONE
            catBreedIdentity.visibility = View.GONE

        }
    }
    private fun setLabels(context: Context){
        //Log.i("setLabels", "entered function")
        val id = context.resources.getIdentifier("label_dictionary","raw", context.packageName)
        val afd = context.resources.openRawResource(id)
        //val afd = context.assets.openFd("labelMap.csv")
        //val input = FileInputStream(afd.fileDescriptor)
        val input = afd
        csvReader().open(input){
            readAllAsSequence().forEach { row ->
                label_map[row[0].toInt()]=row[1]
            }
        }
//        if(label_map.isEmpty()) {
//            Log.i("setLabels", "hashmap is empty")
//        }else{
//            for(key in label_map.keys){
//                Log.i("setLabels","element at key $key : ${label_map.get(key)}")
//            }
//        }
    }

    override fun onResume() {
        super.onResume()
        // onResume is the place to do processing that will need to be active when
        // the activity is shown but doesn't need to be around otherwise. So this
        // is where the background thread gets started and the camera is prepared.
        // If the preview area is already around, the camera can be opened.
        // Otherwise, the preview area's state needs to be observed to react when
        // it does become available.
        openBackgroundThread()
        if (cameraPreview?.isAvailable == true) {
            findCamera()
            openCamera()
        } else {
            cameraPreview?.surfaceTextureListener = surfaceTextureListener
        }
    }

    override fun onStop() {
        super.onStop()
        // onStop is the place that stops everything that was started in onResume.
        // So close the camera and stop the background thread that was using the
        // camera.
        closeCamera()
        closeBackgroundThread()
    }

    private fun setCatIdentity(catid: Array<Float>) {
        // This is the callback that gets called after the predictor has determined
        // the picture's cuteness score. The actions are to display the cuteness score
        // and the judgement of whether that means "cute" or not. Also, the reset button
        // is made visible so that the user can restart the scanning.
        val resetButton = findViewById<Button>(R.id.reset_button)
        val catBreedPercentageText = findViewById<TextView>(R.id.catbreed_percentage)
        val catBreedlabelText = findViewById<TextView>(R.id.catbreed_label)
//        val cutenessPercentageText = findViewById<TextView>(R.id.cuteness_percentage_text)
//        val cutenessEvaluationText = findViewById<TextView>(R.id.cuteness_evaluation_text)

        catBreedPercentageText.text = getString(R.string.catbreed_percentage, 100* catid[1])
        //label_map maps the first index (representing food label) to actual food class name
        catBreedlabelText.text = getString(R.string.catbreed_label, label_map[catid[0].toInt()])
        if (catid[1] <= .5){
            catBreedlabelText.text = "Mixed breed cat"
        }
        resetButton.visibility = View.VISIBLE
        catBreedPercentageText.visibility = View.VISIBLE
        catBreedlabelText.visibility = View.VISIBLE
//        cutenessPercentageText.visibility = View.VISIBLE
//        cutenessEvaluationText.visibility = View.VISIBLE
    }

    private fun openBackgroundThread() {
        // Create and start a new background thread. Make a Handler for it that can be
        // passed to the various camera methods as the processing thread.
        backgroundThread = HandlerThread("catid_camera_thread")
        backgroundThread?.start()
        backgroundHandler = Handler(backgroundThread?.looper)
    }

    private fun closeBackgroundThread() {
        if (backgroundHandler != null) {
            backgroundThread?.quitSafely()
            backgroundThread = null
            backgroundHandler = null
        }
    }

    private fun findCamera() {
        // A phone might have multiple cameras. We want to find the back-facing one,
        // since that's the one that would be used in this kind of app. When it's found
        // the cameraId field is set to that camera's identifier. The cameraId is later
        // used to use the camera.
        for (cameraId in cameraManager?.cameraIdList ?: arrayOf()) {
            val characteristics = cameraManager?.getCameraCharacteristics(cameraId)
            if (characteristics!![CameraCharacteristics.LENS_FACING] == CameraCharacteristics.LENS_FACING_BACK) {
                this.cameraId = cameraId
                break
            }
        }
    }

    private fun openCamera() {
        // To call openCamera on a CameraManager, the camera permission needs to have been
        // granted by the user, so opening the camera needs to be inside the check.
        if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            cameraManager?.openCamera(cameraId, cameraStateCallback, backgroundHandler)
        }
    }

    private fun closeCamera() {
        // Closing the camera means closing the capture session if it was opened, and also
        // closing the camera itself.
        if (cameraCaptureSession != null) {
            cameraCaptureSession?.close()
            cameraCaptureSession = null
        }
        if (camera != null) {
            camera?.close()
            camera = null
        }
    }

    private fun createPreviewSession() {
        // Create a Surface object from the TextureView, and target it with the camera
        // request builder, so that the camera can become active and use the preview
        // area as its display and capture area. TEMPLATE_PREVIEW is used because this
        // is a preview area, and we are not interested in capturing the highest quality
        // possible.
        val texture = cameraPreview!!.surfaceTexture
        val surface = Surface(texture)
        val captureRequestBuilder = camera!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        captureRequestBuilder.addTarget(surface)

        // Create a session with the camera. We need to pass in a list of Surfaces to use.
        // In this simple case, that means only the one Surface that was created above. The
        // callback will be called when the session is ready.
        camera!!.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigureFailed(session: CameraCaptureSession) = Unit

            override fun onConfigured(session: CameraCaptureSession) {
                if (camera == null) {
                    return
                }

                // When the session is ready, it is stored in a field so that it can be closed
                // later when the activity is stopped. Creating a repeating request means that
                // the camera is constantly capturing the image, so when the Evaluate button is
                // pressed, whatever is shown in the preview area is what is available as the
                // captured bitmap of the camera.
                val captureRequest = captureRequestBuilder.build()
                cameraCaptureSession = session
                session.setRepeatingRequest(captureRequest, null, backgroundHandler)
            }

        }, backgroundHandler)
    }

    /**
     * The listener for the TextureView state changes. We are only interested in when the
     * TextureView becomes available, so that reacts by opening the appropriate camera.
     */
    private val surfaceTextureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture?, width: Int, height: Int) = Unit

        override fun onSurfaceTextureUpdated(surface: SurfaceTexture?) = Unit

        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture?) = true

        override fun onSurfaceTextureAvailable(surface: SurfaceTexture?, width: Int, height: Int) {
            findCamera()
            openCamera()
        }

    }

    /**
     * The callback for the camera state. When opening the camera succeeds, a session
     * with the camera needs to be created to be able to capture images.
     */
    private val cameraStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            this@MainActivity.camera = camera
            createPreviewSession()
        }

        override fun onDisconnected(camera: CameraDevice) {
            this@MainActivity.camera?.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            this@MainActivity.camera?.close()
            this@MainActivity.camera = null
        }

    }
}
