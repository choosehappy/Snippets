import static qupath.lib.roi.PathROIToolsAwt.getShape;
import java.awt.image.BufferedImage
import java.awt.Color
import javax.imageio.ImageIO
import qupath.lib.roi.PathROIToolsAwt
import qupath.lib.objects.classes.PathClass
import qupath.lib.common.ColorTools


double downsample = 16.0

// define output directory
def pathOutput = 'c:/brca/' // for windows

def server = getCurrentImageData().getServer()
int w = (server.getWidth() / downsample) as int
int h = (server.getHeight() / downsample) as int
String name = server.getShortServerName()



def annotations = getAnnotationObjects()
def pathClasses = annotations.collect({it.getPathClass()}) as Set


for (pclass in pathClasses){


    def img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)

    // Paint the shapes (this is just 'standard' Java - you might want to modify)
    def g2d = img.createGraphics()
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.setColor(Color.WHITE)


    for (annotation in annotations) {
            if(annotation.getPathClass()!=pclass)
                continue
            def roi = annotation.getROI()
            def shape = PathROIToolsAwt.getShape(roi)

            g2d.fill(shape)

        }


g2d.dispose()

// Save the result

def fileMask = new File(pathOutput, name + '_' +pclass +'_mask.png')
ImageIO.write(img, 'PNG', fileMask)
}



