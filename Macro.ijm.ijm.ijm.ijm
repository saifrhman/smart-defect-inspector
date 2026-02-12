input = getDirectory("Choose INPUT folder (images)");
output = getDirectory("Choose OUTPUT folder (masks)");

setBatchMode(true);

list = getFileList(input);

for (i=0; i<list.length; i++) {
    name = list[i];

    if (endsWith(name, ".jpg") || endsWith(name, ".png") || endsWith(name, ".jpeg")) {

        open(input + name);

        run("8-bit");
        run("Gaussian Blur...", "sigma=2");
        setAutoThreshold("Otsu");
        run("Convert to Mask");

        base = replace(name, ".jpg", "");
        base = replace(base, ".png", "");
        base = replace(base, ".jpeg", "");

        saveAs("PNG", output + base + "_mask.png");
        close();
    }
}

setBatchMode(false);
print("✅ Done. Masks saved to: " + output);
