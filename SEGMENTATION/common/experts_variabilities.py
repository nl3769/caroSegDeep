from functions.utils import GetDir, GetFiles
from functions.LoadDatas import LoadBordersUsingName, LoadAnnotationUsingName, LoadSpatialResolution

import numpy as np
from scipy.io import savemat


class computeExpertsVariabilities():

    def __init__(self,
                 pathToAnnotation,
                 pathToBorders,
                 pathToSequences):

        self.dirAnnotationList = GetDir(pathToAnnotation)
        self.dirAnnotationList.remove('PATHO_LOIZOU_59/')

        self.pathToBorders = pathToBorders
        self.pathToAnnotation = pathToAnnotation
        self.pathToSequences = pathToSequences

        self.resultComputationVariabilities = {}

    def Compute(self):

        for i in range(len(self.dirAnnotationList)):

            nameDataBase = self.dirAnnotationList[i]
            tmpPathAnnotation = self.pathToAnnotation + "/" + nameDataBase
            tmpPathBorders = self.pathToBorders + "/" + nameDataBase

            listAnnotation = GetFiles(tmpPathAnnotation)

            tmpDic = {}

            if (nameDataBase == "HEALTHY_ANDRE_57/"):

                tmpDic["HEALTHY_ANDRE_57"] = {}

                IFC3A1 = np.zeros(0, dtype=np.float32)
                IFC3A2 = np.zeros(0, dtype=np.float32)
                IFC3A3 = np.zeros(0, dtype=np.float32)
                IFC3A4 = np.zeros(0, dtype=np.float32)

                IFC4A1 = np.zeros(0, dtype=np.float32)
                IFC4A2 = np.zeros(0, dtype=np.float32)
                IFC4A3 = np.zeros(0, dtype=np.float32)
                IFC4A4 = np.zeros(0, dtype=np.float32)

                for i in range(len(listAnnotation)):
                    listAnnotation[i] = listAnnotation[i].replace("_A1.mat", "").replace("_A2.mat", "").replace("_A3.mat", "").replace("_A4.mat", "").replace("_IFC3", "").replace("_IFC4", "")

                listAnnotation = list(set(listAnnotation))

                for k in range(len(listAnnotation)):

                        borders = LoadBordersUsingName(borders_path = self.pathToBorders + "/",
                                                       name_database = nameDataBase,
                                                       borders_name = listAnnotation[k] + "_borders.mat")

                        spatialResolution = LoadSpatialResolution(self.pathToSequences + "/" + nameDataBase + listAnnotation[k]) * 10000 #im um

                        annotationA1 = LoadAnnotationUsingName(contours_path = self.pathToAnnotation + "/",
                                                                                    name_database = nameDataBase,
                                                               nameAnnotationIFC3 = listAnnotation[k] + "_IFC3_A1.mat",
                                                               nameAnnotationIFC4 = listAnnotation[k] + "_IFC4_A1.mat")

                        annotationA2 = LoadAnnotationUsingName(contours_path = self.pathToAnnotation + "/",
                                                               name_database=nameDataBase,
                                                               nameAnnotationIFC3 = listAnnotation[k] + "_IFC3_A2.mat",
                                                               nameAnnotationIFC4 = listAnnotation[k] + "_IFC4_A2.mat")

                        annotationA3 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                               name_database=nameDataBase,
                                                               nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A3.mat",
                                                               nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A3.mat")

                        annotationA4 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                               name_database=nameDataBase,
                                                               nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A4.mat",
                                                               nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A4.mat")

                        IFC3A1 = np.concatenate((IFC3A1, spatialResolution*annotationA1["IFC3"][borders[0]-1:borders[1]]))
                        IFC3A2 = np.concatenate((IFC3A2, spatialResolution*annotationA2["IFC3"][borders[0]-1:borders[1]]))
                        IFC3A3 = np.concatenate((IFC3A3, spatialResolution*annotationA3["IFC3"][borders[0]-1:borders[1]]))
                        IFC3A4 = np.concatenate((IFC3A4, spatialResolution*annotationA4["IFC3"][borders[0]-1:borders[1]]))

                        IFC4A1 = np.concatenate((IFC4A1, spatialResolution*annotationA1["IFC4"][borders[0]-1:borders[1]]))
                        IFC4A2 = np.concatenate((IFC4A2, spatialResolution*annotationA2["IFC4"][borders[0]-1:borders[1]]))
                        IFC4A3 = np.concatenate((IFC4A3, spatialResolution*annotationA3["IFC4"][borders[0]-1:borders[1]]))
                        IFC4A4 = np.concatenate((IFC4A4, spatialResolution*annotationA4["IFC4"][borders[0]-1:borders[1]]))

                # ------ Computation of the LI difference
                diffA1A2_IFC3 = np.abs(IFC3A1 - IFC3A2)
                diffA1A3_IFC3 = np.abs(IFC3A1 - IFC3A3)
                diffA2A3_IFC3 = np.abs(IFC3A2 - IFC3A3)

                diffA1A4_IFC3 = np.abs(IFC3A1 - IFC3A4)


                # ------ Computation of the MA difference
                diffA1A2_IFC4 = np.abs(IFC4A1 - IFC4A2)
                diffA1A3_IFC4 = np.abs(IFC4A1 - IFC4A3)
                diffA2A3_IFC4 = np.abs(IFC4A2 - IFC4A3)
                diffA1A4_IFC4 = np.abs(IFC4A1 - IFC4A4)

                # ------ Computation of the IM difference
                diffIM_A1A2 = np.abs(IFC4A1 - IFC3A1 - (IFC4A2 - IFC3A2))
                diffIM_A1A3 = np.abs(IFC4A1 - IFC3A1 - (IFC4A3 - IFC3A3))
                diffIM_A2A3 = np.abs(IFC4A2 - IFC3A2 - (IFC4A3 - IFC3A3))
                diffIM_A1A4 = np.abs(IFC4A1 - IFC3A1 - (IFC4A4 - IFC3A4))

                # #  ------ Median value
                # diffIM_A1A2 = np.abs(np.median(IFC4A1 - IFC3A1 - (IFC4A2 - IFC3A2)))
                # diffIM_A1A3 = np.abs(np.median(IFC4A1 - IFC3A1 - (IFC4A3 - IFC3A3)))
                # diffIM_A2A3 = np.abs(np.median(IFC4A2 - IFC3A2 - (IFC4A3 - IFC3A3)))
                # diffIM_A1A4 = np.abs(np.median(IFC4A1 - IFC3A1 - (IFC4A4 - IFC3A4)))


                # ------ Inter variabilities
                tmpDic["HEALTHY_ANDRE_57"]["InterMAE(LI)"] = np.mean(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))
                tmpDic["HEALTHY_ANDRE_57"]["InterMAE(LI_STD)"] = np.std(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))

                tmpDic["HEALTHY_ANDRE_57"]["InterMAE(MA)"] = np.mean(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))
                tmpDic["HEALTHY_ANDRE_57"]["InterMAE(MA,STD)"] = np.std(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))

                tmpDic["HEALTHY_ANDRE_57"]["InterMAE(IMT)"] = np.mean(np.concatenate((diffIM_A1A2, diffIM_A1A3, diffIM_A2A3)))
                tmpDic["HEALTHY_ANDRE_57"]["InterMAE(IMT,STD)"] = np.std(np.concatenate((diffIM_A1A2, diffIM_A1A3, diffIM_A2A3)))

                # ------ Intra variabilities
                tmpDic["HEALTHY_ANDRE_57"]["IntraMAE(LI)"] = np.mean(diffA1A4_IFC3)
                tmpDic["HEALTHY_ANDRE_57"]["IntraMAE(LI_STD)"] = np.std(diffA1A4_IFC3)

                tmpDic["HEALTHY_ANDRE_57"]["IntraMAE(MA)"] = np.mean(diffA1A4_IFC4)
                tmpDic["HEALTHY_ANDRE_57"]["IntraMAE(MA_STD)"] = np.std(diffA1A4_IFC4)

                tmpDic["HEALTHY_ANDRE_57"]["IntraMAE(IMT)"] = np.mean(diffIM_A1A4)
                tmpDic["HEALTHY_ANDRE_57"]["IntraMAE(IMT_STD)"] = np.std(diffIM_A1A4)

                self.resultComputationVariabilities["HEALTHY_ANDRE_57"] = tmpDic["HEALTHY_ANDRE_57"]

            if (nameDataBase == "PATHO_ANDRE_25/"):

                tmpDic["PATHO_ANDRE_25"] = {}

                IFC3A1 = np.zeros(0, dtype=np.float32)
                IFC3A2 = np.zeros(0, dtype=np.float32)
                IFC3A3 = np.zeros(0, dtype=np.float32)
                IFC3A4 = np.zeros(0, dtype=np.float32)

                IFC4A1 = np.zeros(0, dtype=np.float32)
                IFC4A2 = np.zeros(0, dtype=np.float32)
                IFC4A3 = np.zeros(0, dtype=np.float32)
                IFC4A4 = np.zeros(0, dtype=np.float32)

                for i in range(len(listAnnotation)):
                    listAnnotation[i] = listAnnotation[i].replace("_A1.mat", "").replace("_A2.mat", "").replace("_A3.mat", "").replace("_A4.mat", "").replace("_IFC3", "").replace("_IFC4", "")

                listAnnotation = list(set(listAnnotation))

                for k in range(len(listAnnotation)):
                    borders = LoadBordersUsingName(borders_path=self.pathToBorders + "/",
                                                   name_database=nameDataBase,
                                                   borders_name=listAnnotation[k] + "_borders.mat")

                    spatialResolution = LoadSpatialResolution(self.pathToSequences + "/" + nameDataBase + listAnnotation[k]) * 10000 #im um

                    annotationA1 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A1.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A1.mat")

                    annotationA2 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A2.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A2.mat")

                    annotationA3 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A3.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A3.mat")

                    annotationA4 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A4.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A4.mat")

                    IFC3A1 = np.concatenate((IFC3A1, spatialResolution*annotationA1["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A2 = np.concatenate((IFC3A2, spatialResolution*annotationA2["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A3 = np.concatenate((IFC3A3, spatialResolution*annotationA3["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A4 = np.concatenate((IFC3A4, spatialResolution*annotationA4["IFC3"][borders[0] - 1:borders[1]]))

                    IFC4A1 = np.concatenate((IFC4A1, spatialResolution*annotationA1["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A2 = np.concatenate((IFC4A2, spatialResolution*annotationA2["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A3 = np.concatenate((IFC4A3, spatialResolution*annotationA3["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A4 = np.concatenate((IFC4A4, spatialResolution*annotationA4["IFC4"][borders[0] - 1:borders[1]]))

                # ------ Computation of the LI difference
                diffA1A2_IFC3 = np.abs(IFC3A1 - IFC3A2)
                diffA1A3_IFC3 = np.abs(IFC3A1 - IFC3A3)
                diffA2A3_IFC3 = np.abs(IFC3A2 - IFC3A3)
                diffA1A4_IFC3 = np.abs(IFC3A1 - IFC3A4)

                # ------ Computation of the MA difference
                diffA1A2_IFC4 = np.abs(IFC4A1 - IFC4A2)
                diffA1A3_IFC4 = np.abs(IFC4A1 - IFC4A3)
                diffA2A3_IFC4 = np.abs(IFC4A2 - IFC4A3)
                diffA1A4_IFC4 = np.abs(IFC4A1 - IFC4A4)

                # ------ Computation of the IM difference
                diffIM_A1A2 = np.abs(IFC4A1 - IFC3A1 - (IFC4A2 - IFC3A2))
                diffIM_A1A3 = np.abs(IFC4A1 - IFC3A1 - (IFC4A3 - IFC3A3))
                diffIM_A2A3 = np.abs(IFC4A2 - IFC3A2 - (IFC4A3 - IFC3A3))
                diffIM_A1A4 = np.abs(IFC4A1 - IFC3A1 - (IFC4A4 - IFC3A4))


                # ------ Inter variabilities
                tmpDic["PATHO_ANDRE_25"]["InterMAE(LI)"] = np.mean(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))
                tmpDic["PATHO_ANDRE_25"]["InterMAE(LI_STD)"] = np.std(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))

                tmpDic["PATHO_ANDRE_25"]["InterMAE(MA)"] = np.mean(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))
                tmpDic["PATHO_ANDRE_25"]["InterMAE(MA,STD)"] = np.std(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))

                tmpDic["PATHO_ANDRE_25"]["InterMAE(IMT)"] = np.mean(np.concatenate((diffIM_A1A2, diffIM_A1A3, diffIM_A2A3)))
                tmpDic["PATHO_ANDRE_25"]["InterMAE(IMT,STD)"] = np.std(np.concatenate((diffIM_A1A2, diffIM_A1A3, diffIM_A2A3)))

                # ------ Intra variabilities
                tmpDic["PATHO_ANDRE_25"]["IntraMAE(LI)"] = np.mean(diffA1A4_IFC3)
                tmpDic["PATHO_ANDRE_25"]["IntraMAE(LI_STD)"] = np.std(diffA1A4_IFC3)

                tmpDic["PATHO_ANDRE_25"]["IntraMAE(MA)"] = np.mean(diffA1A4_IFC4)
                tmpDic["PATHO_ANDRE_25"]["IntraMAE(MA_STD)"] = np.std(diffA1A4_IFC4)

                tmpDic["PATHO_ANDRE_25"]["IntraMAE(IMT)"] = np.mean(diffIM_A1A4)
                tmpDic["PATHO_ANDRE_25"]["IntraMAE(IMT_STD)"] = np.std(diffIM_A1A4)

                self.resultComputationVariabilities["PATHO_ANDRE_25"] = tmpDic["PATHO_ANDRE_25"]

            if (nameDataBase == "PATHO_AAD_51/"):

                tmpDic["PATHO_AAD_51"] = {}

                IFC3A1 = np.zeros(0, dtype=np.float32)
                IFC3A2 = np.zeros(0, dtype=np.float32)
                IFC3A3 = np.zeros(0, dtype=np.float32)

                IFC4A1 = np.zeros(0, dtype=np.float32)
                IFC4A2 = np.zeros(0, dtype=np.float32)
                IFC4A3 = np.zeros(0, dtype=np.float32)

                for i in range(len(listAnnotation)):
                    listAnnotation[i] = listAnnotation[i].replace("_A1.mat", "").replace("_A2.mat", "").replace("_A3.mat", "").replace("_A4.mat", "").replace("_IFC3", "").replace("_IFC4", "")

                listAnnotation = list(set(listAnnotation))

                for k in range(len(listAnnotation)):
                    borders = LoadBordersUsingName(borders_path=self.pathToBorders + "/",
                                                   name_database=nameDataBase,
                                                   borders_name=listAnnotation[k] + "_borders.mat")

                    spatialResolution = LoadSpatialResolution(self.pathToSequences + "/" + nameDataBase + listAnnotation[k]) * 10000 #im um

                    annotationA1 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A1.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A1.mat")

                    annotationA2 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A2.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A2.mat")

                    annotationA3 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A3.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A3.mat")


                    IFC3A1 = np.concatenate((IFC3A1, spatialResolution*annotationA1["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A2 = np.concatenate((IFC3A2, spatialResolution*annotationA2["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A3 = np.concatenate((IFC3A3, spatialResolution*annotationA3["IFC3"][borders[0] - 1:borders[1]]))

                    IFC4A1 = np.concatenate((IFC4A1, spatialResolution*annotationA1["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A2 = np.concatenate((IFC4A2, spatialResolution*annotationA2["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A3 = np.concatenate((IFC4A3, spatialResolution*annotationA3["IFC4"][borders[0] - 1:borders[1]]))

                # ------ Computation of the LI difference
                diffA1A2_IFC3 = np.abs(IFC3A1 - IFC3A2)
                diffA1A3_IFC3 = np.abs(IFC3A1 - IFC3A3)
                diffA2A3_IFC3 = np.abs(IFC3A2 - IFC3A3)

                # ------ Computation of the MA difference
                diffA1A2_IFC4 = np.abs(IFC4A1 - IFC4A2)
                diffA1A3_IFC4 = np.abs(IFC4A1 - IFC4A3)
                diffA2A3_IFC4 = np.abs(IFC4A2 - IFC4A3)

                # ------ Computation of the IM difference
                diffIM_A1A2 = np.abs(IFC4A1 - IFC3A1 - (IFC4A2 - IFC3A2))
                diffIM_A1A3 = np.abs(IFC4A1 - IFC3A1 - (IFC4A3 - IFC3A3))
                diffIM_A2A3 = np.abs(IFC4A2 - IFC3A2 - (IFC4A3 - IFC3A3))


                # ------ Inter variabilities
                tmpDic["PATHO_AAD_51"]["InterMAE(LI)"] = np.mean(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))
                tmpDic["PATHO_AAD_51"]["InterMAE(LI_STD)"] = np.std(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))

                tmpDic["PATHO_AAD_51"]["InterMAE(MA)"] = np.mean(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))
                tmpDic["PATHO_AAD_51"]["InterMAE(MA,STD)"] = np.std(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))

                tmpDic["PATHO_AAD_51"]["InterMAE(IMT)"] = np.mean(np.concatenate((diffIM_A1A2, diffIM_A1A3, diffIM_A2A3)))
                tmpDic["PATHO_AAD_51"]["InterMAE(IMT,STD)"] = np.std(np.concatenate((diffIM_A1A2, diffIM_A1A3, diffIM_A2A3)))


                self.resultComputationVariabilities["PATHO_AAD_51"] = tmpDic["PATHO_AAD_51"]

            if (nameDataBase == "HEALTHY_MICHAEL_30/"):

                tmpDic["HEALTHY_MICHAEL_30"] = {}

                IFC3A1 = np.zeros(0, dtype=np.float32)
                IFC3A2 = np.zeros(0, dtype=np.float32)
                IFC3A3 = np.zeros(0, dtype=np.float32)

                IFC4A1 = np.zeros(0, dtype=np.float32)
                IFC4A2 = np.zeros(0, dtype=np.float32)
                IFC4A3 = np.zeros(0, dtype=np.float32)

                IMTA1 = []
                IMTA2 = []
                IMTA3 = []
                # IMTA1 = []
                for i in range(len(listAnnotation)):
                    listAnnotation[i] = listAnnotation[i].replace("_A1.mat", "").replace("_A2.mat", "").replace("_A3.mat", "").replace("_A4.mat", "").replace("_IFC3", "").replace("_IFC4", "")

                listAnnotation = list(set(listAnnotation))

                for k in range(len(listAnnotation)):
                    borders = LoadBordersUsingName(borders_path=self.pathToBorders + "/",
                                                   name_database=nameDataBase,
                                                   borders_name=listAnnotation[k] + "_borders.mat")

                    spatialResolution = LoadSpatialResolution(self.pathToSequences + "/" + nameDataBase + listAnnotation[k]) * 10000 #im um

                    annotationA1 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A1.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A1.mat")

                    annotationA2 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A2.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A2.mat")

                    annotationA3 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A3.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A3.mat")


                    IMTA1.append(np.median(spatialResolution*annotationA1["IFC4"][borders[0] - 1:borders[1]]-spatialResolution*annotationA1["IFC3"][borders[0] - 1:borders[1]]))
                    IMTA2.append(np.median(spatialResolution*annotationA2["IFC4"][borders[0] - 1:borders[1]]-spatialResolution*annotationA2["IFC3"][borders[0] - 1:borders[1]]))
                    IMTA3.append(np.median(spatialResolution*annotationA3["IFC4"][borders[0] - 1:borders[1]]-spatialResolution*annotationA3["IFC3"][borders[0] - 1:borders[1]]))

                    # LIA1med.append(np.median(spatialResolution*annotationA1["IFC3"][borders[0] - 1:borders[1]]))
                    # MAA1med.append(np.median(spatialResolution * annotationA1["IFC4"][borders[0] - 1:borders[1]]))
                    #
                    # LIA2med.append(np.median(spatialResolution * annotationA2["IFC3"][borders[0] - 1:borders[1]]))
                    # MAA2med.append(np.median(spatialResolution * annotationA2["IFC4"][borders[0] - 1:borders[1]]))
                    #
                    # LIA3med.append(np.median(spatialResolution * annotationA3["IFC3"][borders[0] - 1:borders[1]]))
                    # MAA3med.append(np.median(spatialResolution * annotationA3["IFC4"][borders[0] - 1:borders[1]]))


                    # IMTA4
                    IFC3A1 = np.concatenate((IFC3A1, spatialResolution*annotationA1["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A2 = np.concatenate((IFC3A2, spatialResolution*annotationA2["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A3 = np.concatenate((IFC3A3, spatialResolution*annotationA3["IFC3"][borders[0] - 1:borders[1]]))

                    IFC4A1 = np.concatenate((IFC4A1, spatialResolution*annotationA1["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A2 = np.concatenate((IFC4A2, spatialResolution*annotationA2["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A3 = np.concatenate((IFC4A3, spatialResolution*annotationA3["IFC4"][borders[0] - 1:borders[1]]))

                # ------ Computation of the LI difference
                diffA1A2_IFC3 = np.abs(IFC3A1 - IFC3A2)
                diffA1A3_IFC3 = np.abs(IFC3A1 - IFC3A3)
                diffA2A3_IFC3 = np.abs(IFC3A2 - IFC3A3)

                # ------ Computation of the MA difference
                diffA1A2_IFC4 = np.abs(IFC4A1 - IFC4A2)
                diffA1A3_IFC4 = np.abs(IFC4A1 - IFC4A3)
                diffA2A3_IFC4 = np.abs(IFC4A2 - IFC4A3)

                # ------ Computation of the IM difference
                tmpIMTA1A2 = np.abs(np.asarray(IMTA1)-np.asarray(IMTA2))
                tmpIMTA1A3 = np.abs(np.asarray(IMTA1) - np.asarray(IMTA3))
                tmpIMTA2A3 = np.abs(np.asarray(IMTA2) - np.asarray(IMTA3))
                # err = np.mean(np.concatenate((tmpIMTA1A3, tmpIMTA1A2, tmpIMTA2A3)))
                #####################################################################################################################################################################

                # ------ Inter variabilities
                tmpDic["HEALTHY_MICHAEL_30"]["InterMAE(LI)"] = np.mean(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))
                tmpDic["HEALTHY_MICHAEL_30"]["InterMAE(LI_STD)"] = np.std(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))

                tmpDic["HEALTHY_MICHAEL_30"]["InterMAE(MA)"] = np.mean(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))
                tmpDic["HEALTHY_MICHAEL_30"]["InterMAE(MA,STD)"] = np.std(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))

                tmpDic["HEALTHY_MICHAEL_30"]["InterMAE(IMT)"] = np.mean(np.concatenate((tmpIMTA1A3, tmpIMTA1A2, tmpIMTA2A3)))
                tmpDic["HEALTHY_MICHAEL_30"]["InterMAE(IMT,STD)"] = np.std(np.concatenate((tmpIMTA1A3, tmpIMTA1A2, tmpIMTA2A3)))

                self.resultComputationVariabilities["HEALTHY_MICHAEL_30"] = tmpDic["HEALTHY_MICHAEL_30"]

            if (nameDataBase == "PATHO_MICHAEL_30/"):

                tmpDic["PATHO_MICHAEL_30"] = {}

                IFC3A1 = np.zeros(0, dtype=np.float32)
                IFC3A2 = np.zeros(0, dtype=np.float32)
                IFC3A3 = np.zeros(0, dtype=np.float32)

                IFC4A1 = np.zeros(0, dtype=np.float32)
                IFC4A2 = np.zeros(0, dtype=np.float32)
                IFC4A3 = np.zeros(0, dtype=np.float32)

                IMTA1 = []
                IMTA2 = []
                IMTA3 = []

                for i in range(len(listAnnotation)):
                    listAnnotation[i] = listAnnotation[i].replace("_A1.mat", "").replace("_A2.mat", "").replace("_A3.mat", "").replace("_A4.mat", "").replace("_IFC3", "").replace("_IFC4", "")

                listAnnotation = list(set(listAnnotation))

                for k in range(len(listAnnotation)):
                    borders = LoadBordersUsingName(borders_path=self.pathToBorders + "/",
                                                   name_database=nameDataBase,
                                                   borders_name=listAnnotation[k] + "_borders.mat")
                    spatialResolution = LoadSpatialResolution(self.pathToSequences + "/" + nameDataBase + listAnnotation[k]) * 10000 #im um

                    annotationA1 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A1.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A1.mat")

                    annotationA2 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A2.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A2.mat")

                    annotationA3 = LoadAnnotationUsingName(contours_path=self.pathToAnnotation + "/",
                                                           name_database=nameDataBase,
                                                           nameAnnotationIFC3=listAnnotation[k] + "_IFC3_A3.mat",
                                                           nameAnnotationIFC4=listAnnotation[k] + "_IFC4_A3.mat")


                    IFC3A1 = np.concatenate((IFC3A1, spatialResolution*annotationA1["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A2 = np.concatenate((IFC3A2, spatialResolution*annotationA2["IFC3"][borders[0] - 1:borders[1]]))
                    IFC3A3 = np.concatenate((IFC3A3, spatialResolution*annotationA3["IFC3"][borders[0] - 1:borders[1]]))

                    IFC4A1 = np.concatenate((IFC4A1, spatialResolution*annotationA1["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A2 = np.concatenate((IFC4A2, spatialResolution*annotationA2["IFC4"][borders[0] - 1:borders[1]]))
                    IFC4A3 = np.concatenate((IFC4A3, spatialResolution*annotationA3["IFC4"][borders[0] - 1:borders[1]]))

                    IMTA1.append(np.median(spatialResolution*annotationA1["IFC4"][borders[0] - 1:borders[1]]-spatialResolution*annotationA1["IFC3"][borders[0] - 1:borders[1]]))
                    IMTA2.append(np.median(spatialResolution*annotationA2["IFC4"][borders[0] - 1:borders[1]]-spatialResolution*annotationA2["IFC3"][borders[0] - 1:borders[1]]))
                    IMTA3.append(np.median(spatialResolution*annotationA3["IFC4"][borders[0] - 1:borders[1]]-spatialResolution*annotationA3["IFC3"][borders[0] - 1:borders[1]]))

                # ------ Computation of the LI difference
                diffA1A2_IFC3 = np.abs(IFC3A1 - IFC3A2)
                diffA1A3_IFC3 = np.abs(IFC3A1 - IFC3A3)
                diffA2A3_IFC3 = np.abs(IFC3A2 - IFC3A3)

                # ------ Computation of the LI difference
                diffA1A2_IFC4 = np.abs(IFC4A1 - IFC4A2)
                diffA1A3_IFC4 = np.abs(IFC4A1 - IFC4A3)
                diffA2A3_IFC4 = np.abs(IFC4A2 - IFC4A3)

                # ------ Computation of the IM difference
                tmpIMTA1A2 = np.abs(np.asarray(IMTA1) - np.asarray(IMTA2))
                tmpIMTA1A3 = np.abs(np.asarray(IMTA1) - np.asarray(IMTA3))
                tmpIMTA2A3 = np.abs(np.asarray(IMTA2) - np.asarray(IMTA3))

                # ------ Inter variabilities
                tmpDic["PATHO_MICHAEL_30"]["InterMAE(LI)"] = np.mean(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))
                tmpDic["PATHO_MICHAEL_30"]["InterMAE(LI_STD)"] = np.std(np.concatenate((diffA1A2_IFC3, diffA1A3_IFC3, diffA2A3_IFC3)))

                tmpDic["PATHO_MICHAEL_30"]["InterMAE(MA)"] = np.mean(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))
                tmpDic["PATHO_MICHAEL_30"]["InterMAE(MA,STD)"] = np.std(np.concatenate((diffA1A2_IFC4, diffA1A3_IFC4, diffA2A3_IFC4)))

                tmpDic["PATHO_MICHAEL_30"]["InterMAE(IMT)"] = np.mean(np.concatenate((tmpIMTA1A3, tmpIMTA1A2, tmpIMTA2A3)))
                tmpDic["PATHO_MICHAEL_30"]["InterMAE(IMT,STD)"] = np.std(np.concatenate((tmpIMTA1A3, tmpIMTA1A2, tmpIMTA2A3)))

                self.resultComputationVariabilities["PATHO_MICHAEL_30"] = tmpDic["PATHO_MICHAEL_30"]


        savemat("logs/expertsVariabilities.mat", self.resultComputationVariabilities)
        print("Results are saved in logs/")
