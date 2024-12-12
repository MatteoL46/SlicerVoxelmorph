import logging
import os
from typing import Annotated, Optional

import vtk, ctk, qt


import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import SimpleITK as sitk
import sitkUtils

import tempfile


#from slicer import vtkMRMLScalarVolumeNode


#
# Voxelmorph
#


class Voxelmorph(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Voxelmorph")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Registration")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Matteo Latella (Magna Graecia University of Catanzaro, Italy)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Voxelmorph">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This module is based on the Matteo Latella's master thesis. Voxelmorph was developed by Dr. Andrian Dalca and his team.
""")


#
# Register sample data sets in Sample Data module
#


#
# VoxelmorphWidget
#


class VoxelmorphWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def setup(self):
      ScriptedLoadableModuleWidget.setup(self)

      # Instantiate and connect widgets ...

      #
      # Parameters Area
      #
      parametersCollapsibleButton = ctk.ctkCollapsibleButton()
      parametersCollapsibleButton.text = "Parameters"
      self.layout.addWidget(parametersCollapsibleButton)

      # Layout within the dummy collapsible button
      parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

      #
      # Fixed volume selector
      #
      self.FixedSelector = slicer.qMRMLNodeComboBox()
      self.FixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
      self.FixedSelector.selectNodeUponCreation = True
      self.FixedSelector.addEnabled = False
      self.FixedSelector.removeEnabled = False
      self.FixedSelector.noneEnabled = False
      self.FixedSelector.showHidden = False
      self.FixedSelector.showChildNodeTypes = False
      self.FixedSelector.setMRMLScene(slicer.mrmlScene)
      self.FixedSelector.setToolTip( "Select the fixed image" )
      parametersFormLayout.addRow("Fixed image: ", self.FixedSelector)
      


      #
      # Moving volume selector
      #
      self.MovingSelector = slicer.qMRMLNodeComboBox()
      self.MovingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
      self.MovingSelector.selectNodeUponCreation = True
      self.MovingSelector.addEnabled = False
      self.MovingSelector.removeEnabled = False
      self.MovingSelector.noneEnabled = False
      self.MovingSelector.showHidden = False
      self.MovingSelector.showChildNodeTypes = False
      self.MovingSelector.setMRMLScene(slicer.mrmlScene)
      self.MovingSelector.setToolTip( "Select the moving image" )
      parametersFormLayout.addRow("Moving image: ", self.MovingSelector)


      #
      # output volume selector
      #

      self.registrationOutputSelector = slicer.qMRMLNodeComboBox()
      self.registrationOutputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
      self.registrationOutputSelector.selectNodeUponCreation = True
      self.registrationOutputSelector.addEnabled = True
      self.registrationOutputSelector.removeEnabled = True
      self.registrationOutputSelector.noneEnabled = True
      self.registrationOutputSelector.showHidden = False
      self.registrationOutputSelector.showChildNodeTypes = False
      self.registrationOutputSelector.setMRMLScene(slicer.mrmlScene)
      self.registrationOutputSelector.baseName = "Registration"
      self.registrationOutputSelector.setToolTip("Select or create a registration")
      parametersFormLayout.addRow("Output registration: ", self.registrationOutputSelector)

      #
      # Apply Button
      #
      self.applyButton = qt.QPushButton("Apply (it can take some minutes)")
      self.applyButton.toolTip = "Run the algorithm."
      self.applyButton.enabled = False
      parametersFormLayout.addRow(self.applyButton)

      # connections
      self.applyButton.connect('clicked(bool)', self.onApplyButton)
      self.FixedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
      self.MovingSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
      self.registrationOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

      # Add vertical spacer
      self.layout.addStretch(1)

      # Refresh Apply button state
      self.onSelect()

      # Create logic object
      self.logic = VoxelmorphLogic()

    def onSelect(self):
       self.applyButton.enabled = self.FixedSelector.currentNode(), self.MovingSelector.currentNode() and self.registrationOutputSelector.currentNode()


    def onApplyButton(self):
        self.logic.run(self.FixedSelector.currentNode(), self.MovingSelector.currentNode(), self.registrationOutputSelector.currentNode())


#
# VoxelmorphLogic
#


class VoxelmorphLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    

    def run(self, FixedNode, MovingNode, registrationOutputNode):
        
        try:
            import os
            import numpy as np
            import voxelmorph as vxm
            import tensorflow as tf
            import tempfile
            import SimpleITK as sitk
            import sitkUtils
            from skimage.transform import resize
            import time

        except:
            print("Please  wait, I am installing all the required libraries. It can be take a while!")
            slicer.util.pip_install('https://github.com/adalca/neurite/archive/refs/heads/dev.zip')
            slicer.util.pip_install('https://github.com/voxelmorph/voxelmorph/archive/refs/heads/dev.zip')
            slicer.util.pip_install('tensorflow')

            import os
            import numpy as np
            import voxelmorph as vxm
            import tensorflow as tf
            import tempfile
            import SimpleITK as sitk
            import sitkUtils
            from skimage.transform import resize

        intial_time = time.time()

        # tensorflow device handling
        device, nb_devices = vxm.tf.utils.setup_device()

        #import pbd; pdb.set_trace()

        # Convert Slicer volume to SimpleITK
        Fixed_sitk = sitk.Cast(sitkUtils.PullVolumeFromSlicer(FixedNode.GetName()), sitk.sitkFloat32)
        fixed_np = sitk.GetArrayFromImage(Fixed_sitk)

        # Convert Slicer volume to SimpleITK
        Moving_sitk = sitk.Cast(sitkUtils.PullVolumeFromSlicer(MovingNode.GetName()), sitk.sitkFloat32)
        moving_np = sitk.GetArrayFromImage(Moving_sitk)
        moving_np = resize(moving_np, fixed_np.shape, order = 1)

        # Convert Slicer volume to SimpleITK
        #registrationOutput_sitk = sitk.Cast(sitkUtils.PullVolumeFromSlicer(registrationOutputNode.GetName()), sitk.sitkFloat32)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as moving_tempfile, \
            tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as fixed_tempfile, \
            tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as outputRegistration_tempfile:

            #Save the moving image temporarily
            np.savez(moving_tempfile.name, vol=moving_np)
            #np.savez("C:\\Users\\Matteo\Desktop\\Data_nostro\\MNI152_T1_1mm_brain_norm.npz", vol=moving_np)


            #Save the fixed image temporarily
            np.savez(fixed_tempfile.name, vol=fixed_np)
            #np.savez("C:\\Users\\Matteo\\Desktop\\Data_nostro\\Tizio_T1_norm.npz", vol=fixed_np)


            # Assume fixed image is already in a known location
            fixed_image_path = fixed_tempfile.name
            #fixed_image_path = "C:\\Users\\Matteo\\Desktop\\voxelmorph-dev\\voxelmorph-dev\\data\\atlas.npz"

            # Assume Moving image is already in a known location
            moving_image_path = moving_tempfile.name
            #moving_image_path = "C:\\Users\\Matteo\\Desktop\\voxelmorph-dev\\voxelmorph-dev\\data\\test_scan.npz"

            outputRegistration_image_path = outputRegistration_tempfile.name
            
            # Load the moving and fixed images
            add_feat_axis = True
            moving = vxm.py.utils.load_volfile(moving_image_path, add_batch_axis=True, add_feat_axis=add_feat_axis)
            fixed, fixed_affine = vxm.py.utils.load_volfile(fixed_image_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

            #import pdb; pdb.set_trace()

            inshape = moving.shape[1:-1]
            nb_feats = moving.shape[-1]

            # Device handling and model prediction
            with tf.device(device):
                config = dict(inshape=inshape, input_model=None)
                model_fn = __file__.replace("Voxelmorph.py", "Resources" + os.sep + "vxm_dense_brain_T1_3D_mse.h5")
                warp = vxm.networks.VxmDense.load(model_fn, **config).register(moving, fixed)
                moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

            # Save the moved image
            vxm.py.utils.save_volfile(moved.squeeze(), outputRegistration_image_path, fixed_affine)
            vxm.py.utils.save_volfile(moved.squeeze(), "C:\\Users\\Matteo\\Desktop\\Data_nostro\\out_slicer2.nii.gz", fixed_affine)

            #import pdb; pdb.set_trace()
             
            # Read the output image and push it back to Slicer
            #outputRegistration_sitk = sitk.ReadImage(outputRegistration_image_path)
            outputRegistration_sitk = sitk.PermuteAxes(sitk.ReadImage(outputRegistration_image_path), (2,1,0))
            outputRegistration_sitk.CopyInformation(Fixed_sitk)
            sitk.WriteImage(outputRegistration_sitk, "C:\\Users\\Matteo\\Desktop\\Data_nostro\\out_slicer3.nii.gz")

            # Create and push the labelmap to Slicer
            registrationOutputNode = sitkUtils.PushVolumeToSlicer(outputRegistration_sitk, registrationOutputNode)
            slicer.util.setSliceViewerLayers(background=registrationOutputNode)
            #displayNode=registrationOutputNode.GetDisplayNode()
            
        print(time.time() - intial_time)


        # Cleanup temporary files
        #os.remove(moving_tempfile.name)
        #os.remove(fixed_tempfile.name)
        #os.remove(outputRegistration_tempfile.name)


#
# VoxelmorphTest
#


class VoxelmorphTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_Voxelmorph1()

    def test_Voxelmorph1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

