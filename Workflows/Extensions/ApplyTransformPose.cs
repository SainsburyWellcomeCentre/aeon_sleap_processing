using Bonsai;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;

[Combinator]
[Description("Applies a homography transformation to the centroids in a PoseIdentityCollection.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ApplyTransformPose
{
    public double[] HomographyMatrix { get; set; }

    public IObservable<PoseIdentityCollection> Process(IObservable<PoseIdentityCollection> source)
    {
        return source.Select(value =>
        {
            var inputPoseIdentities = value;
            var outputCollection = new PoseIdentityCollection(inputPoseIdentities.Image, inputPoseIdentities.Model);

            foreach (var identity in inputPoseIdentities)
            {
                // Transform the centroid position using the homography matrix
                var transformedCentroidPosition = TransformPoint(identity.Centroid.Position, HomographyMatrix);

                // Create a new BodyPart for the transformed centroid
                var transformedCentroid = new BodyPart
                {
                    Position = transformedCentroidPosition,
                    Confidence = identity.Centroid.Confidence // Retain the original confidence value
                };

                // Create a new PoseIdentity with the transformed centroid
                var transformedIdentity = new PoseIdentity(identity.Image, identity.Model)
                {
                    Centroid = transformedCentroid,

                    Identity = identity.Identity,
                    IdentityIndex = identity.IdentityIndex,
                    Confidence = identity.Confidence,
                    IdentityScores = identity.IdentityScores
                };

                // Add the transformed identity to the output collection
                outputCollection.Add(transformedIdentity);
            }

            return outputCollection;
        });
    }

    private Point2f TransformPoint(Point2f point, double[] homographyMatrix)
    {
        float x = point.X;
        float y = point.Y;

        // Map the linear array
        double a00 = homographyMatrix[0];
        double a01 = homographyMatrix[1];
        double a02 = homographyMatrix[2];
        double a10 = homographyMatrix[3];
        double a11 = homographyMatrix[4];
        double a12 = homographyMatrix[5];
        double a20 = homographyMatrix[6];
        double a21 = homographyMatrix[7];
        double a22 = homographyMatrix[8];

        // Apply the homography transform
        double tx = (a00 * x) + (a01 * y) + a02;
        double ty = (a10 * x) + (a11 * y) + a12;
        double tw = (a20 * x) + (a21 * y) + a22;

        float transformedX = (float)(tx / tw);
        float transformedY = (float)(ty / tw);
        return new Point2f(transformedX, transformedY);
    }
}