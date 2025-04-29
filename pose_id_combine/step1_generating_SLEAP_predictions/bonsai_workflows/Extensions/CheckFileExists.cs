using Bonsai;
using System;
using System.ComponentModel;
using System.IO;
using System.Reactive;
using System.Reactive.Linq;
using System.Runtime.CompilerServices;

[Combinator]
[Description("Checks whether the file at the specified path exists and returns a boolean value.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CheckFileExists
{
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string FileName { get; set; }

    public IObservable<bool> Process()
    {
        return Observable.Defer(() => Observable.Return(File.Exists(FileName)));
    }

    public IObservable<bool> Process<TSource>(IObservable<TSource> source)
    {
        return source.Select(input =>
        {
            var path = input as string;
            if (path == null)
            {
                throw new InvalidOperationException("Input must be a string.");
            }
            return File.Exists(path);
        });
    }
}