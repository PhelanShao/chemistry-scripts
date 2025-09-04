#!/usr/bin/env python3
"""
Test enhanced XYZ converter on specific problematic files
"""

import csv
import os
from pathlib import Path

# Import from the enhanced script
from enhanced_xyz_to_svg import mol_from_xyz_with_bonds, draw_svg
from rdkit import Chem
from rdkit.Chem import inchi

def test_specific_files():
    """Test specific problematic files"""
    print("🧪 Testing Enhanced XYZ Converter on Specific Files")
    print("=" * 60)
    
    # Test files that were previously problematic
    test_files = [
        "MR_904_1_R.xyz",
        "MR_904_1_TS.xyz", 
        "MR_910_2_P.xyz",
        "1/p4f2p61f1p76p33f2p87/P/MR_1_p4f2p61f1p76p33f2p87_P.xyz",
        "1/p4f2p61f1p76p33f2p87/TS/MR_1_p4f2p61f1p76p33f2p87_TS.xyz"
    ]
    
    # Create output directory
    out_dir = Path("test_enhanced_specific")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # QC report
    qc_path = out_dir / "qc_report.csv"
    
    results = []
    
    with open(qc_path, 'w', newline='', encoding='utf-8') as qc_file:
        qc_writer = csv.writer(qc_file)
        qc_writer.writerow(["file", "status", "smiles", "inchikey", "strategy", "note"])
        
        for i, test_file in enumerate(test_files, 1):
            print(f"\n📄 [{i}/{len(test_files)}] Testing: {test_file}")
            
            if not os.path.exists(test_file):
                print(f"   ❌ File not found: {test_file}")
                qc_writer.writerow([test_file, "NOT_FOUND", "", "", "", "File not found"])
                results.append({"file": test_file, "status": "NOT_FOUND"})
                continue
            
            # Test molecule creation
            mol, error, strategy = mol_from_xyz_with_bonds(test_file)
            
            if mol is None:
                print(f"   ❌ Failed to create molecule ({strategy}): {error}")
                qc_writer.writerow([test_file, "FAIL", "", "", strategy, str(error)])
                results.append({"file": test_file, "status": "FAIL", "strategy": strategy})
                continue
            
            try:
                # Generate SMILES and InChIKey
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                try:
                    inchi_key = inchi.MolToInchiKey(mol)
                except:
                    inchi_key = ""
                
                print(f"   ✅ Molecule created ({strategy})")
                print(f"   📝 SMILES: {smiles}")
                if inchi_key:
                    print(f"   🔑 InChIKey: {inchi_key}")
                
                # Save SDF
                file_stem = Path(test_file).stem
                sdf_file = out_dir / (file_stem + ".sdf")
                Chem.MolToMolFile(mol, str(sdf_file))
                
                # Save SMILES
                smiles_file = out_dir / (file_stem + ".smiles")
                smiles_file.write_text(smiles, encoding="utf-8")
                
                # Generate SVG
                svg, svg_error = draw_svg(mol, w=640, h=480, legend=file_stem)
                
                if svg is None:
                    print(f"   ⚠️  SVG generation failed: {svg_error}")
                    qc_writer.writerow([test_file, "SVG_FAIL", smiles, inchi_key, strategy, str(svg_error)])
                    results.append({"file": test_file, "status": "SVG_FAIL", "strategy": strategy, "smiles": smiles})
                else:
                    svg_file = out_dir / (file_stem + ".svg")
                    svg_file.write_text(svg, encoding="utf-8")
                    print(f"   ✅ SVG generated successfully")
                    qc_writer.writerow([test_file, "SUCCESS", smiles, inchi_key, strategy, ""])
                    results.append({"file": test_file, "status": "SUCCESS", "strategy": strategy, "smiles": smiles})
                
            except Exception as e:
                print(f"   ❌ Processing error: {str(e)}")
                qc_writer.writerow([test_file, "ERROR", "", "", strategy, str(e)])
                results.append({"file": test_file, "status": "ERROR", "strategy": strategy})
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary")
    print("=" * 80)
    
    status_counts = {}
    strategy_counts = {}
    
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        
        if "strategy" in result:
            strategy = result["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"📈 Status Distribution:")
    for status, count in status_counts.items():
        percentage = (count / len(results)) * 100
        print(f"   {status}: {count} ({percentage:.1f}%)")
    
    if strategy_counts:
        print(f"\n🔧 Strategy Usage:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(strategy_counts.values())) * 100
            print(f"   {strategy}: {count} ({percentage:.1f}%)")
    
    print(f"\n📁 Output files saved to: {out_dir}")
    print(f"📋 QC report: {qc_path}")
    
    # Show successful molecules
    successful = [r for r in results if r["status"] == "SUCCESS"]
    if successful:
        print(f"\n✅ Successfully processed molecules:")
        for result in successful:
            print(f"   {Path(result['file']).name} ({result['strategy']}): {result['smiles']}")
    
    print("=" * 80)

if __name__ == "__main__":
    test_specific_files()
